import torch
from utils.NS_Solver_vorticity import solver_ns        
from models.DualEnhance.UQ_class import UQ_model

# dual enhance class 
class dual_enhance():
    def __init__(self, modelConfig):
        self.modelConfig = modelConfig
        self.UQ_model = UQ_model(modelConfig)
        self.S = self.modelConfig['S']
        self.T = self.modelConfig['T']
        self.T_in = self.modelConfig['T_in']
    
    def main_model_inference(self, dd_model, p_model, x_adv):
        # datadriven model inference
        out_datadriven = dd_model.model(x_adv).view(x_adv.shape[0], self.S, self.S, self.T)
        out_datadriven_orig = self.UQ_model.data_dict['y_normalizer_cuda'].decode(out_datadriven)
        # physics solver inference        
        x_orig = self.UQ_model.data_dict['a_normalizer_cuda'].decode(x_adv)
        out_solver = solver_ns(x_orig[:,:,:,-1], visc=self.modelConfig['visc'], batch_size_first=True, Solver_T=self.T, delta_t=2e-3).squeeze()
        out_solver_n = self.UQ_model.data_dict['y_normalizer_cuda'].encode(out_solver.view(x_adv.shape[0], self.S, self.S, self.T)).detach()
        # physics model inference        
        out_physics = p_model.model(x_adv, out_solver_n).view(x_adv.shape[0], self.S, self.S, self.T) + out_solver_n
        out_physics_orig = self.UQ_model.data_dict['y_normalizer_cuda'].decode(out_physics)
        return out_datadriven, out_datadriven_orig, out_physics, out_physics_orig
    
    def dual_enhance(self, dd_model, p_model):
        dd_model.model.train()
        p_model.model.train()
        generation_stop_index = 0
        opt_dd_sum = 0
        opt_p_sum = 0
        train_l2_dual_dd = 0
        train_l2_dual_p = 0
        for x, y_solver, y in self.UQ_model.data_dict['train_loader']:
            generation_stop_index += 1
            if generation_stop_index > self.modelConfig['dual_num']:
                break
            # poor data generate
            x_adv = self.data_generator(x, y, dd_model, p_model)
            y = self.UQ_model.data_dict['y_normalizer_cuda'].decode(y.cuda())
            for dual_opt_index in range(self.modelConfig['dual_opt_repeat']):
                # main model inference
                out_datadriven, out_datadriven_orig, out_physics, out_physics_orig = self.main_model_inference(dd_model, p_model, x_adv)
                # evaluate_uncertainty
                dd_loss = self.UQ_model.myloss(out_datadriven_orig.view(y.shape[0],-1), y.view(y.shape[0],-1)) / y.shape[0]
                p_loss = self.UQ_model.myloss(out_physics_orig.view(y.shape[0],-1), y.view(y.shape[0],-1)) / y.shape[0]
                print("batch_index",generation_stop_index, "epoch", dual_opt_index, "dd_l2_loss:", dd_loss.item(), "p_l2_loss:", p_loss.item())
                out_UQ_dp = self.UQ_model.evaluate_uncertainty(x_adv, out_datadriven, out_physics)
                # dual enhance -- divide training data
                if self.modelConfig['physics_single_t']:
                    out_UQ_dp = torch.ones_like(out_UQ_dp)
                if dual_opt_index == 0:
                    opt_dd_num = out_UQ_dp[out_UQ_dp > 0].shape[0]
                    opt_p_num = out_UQ_dp[out_UQ_dp < 0].shape[0]
                else:
                    if (opt_dd_num != out_UQ_dp[out_UQ_dp > 0].shape[0]) or (opt_p_num != out_UQ_dp[out_UQ_dp < 0].shape[0]):
                        break
                # dual enhance -- optimization
                if opt_dd_num != 0:
                    # opt data driven model
                    if self.modelConfig['use_y']:
                        out_physics_orig = y
                    l2_datadriven = self.UQ_model.myloss(out_datadriven_orig[out_UQ_dp>0,...].view(opt_dd_num,-1), 
                                                out_physics_orig[out_UQ_dp>0,...].view(opt_dd_num,-1))
                    dd_model.optimizer.zero_grad()
                    l2_datadriven.backward()
                    dd_model.optimizer.step()
                    train_l2_dual_dd += l2_datadriven.item()
                    # print("batch_index",generation_stop_index, "epoch", dual_opt_index, "opt_dd_l2_loss:", l2_datadriven.item()/opt_dd_num)
                if opt_p_num != 0:
                    # opt physics model
                    if self.modelConfig['use_y']:
                        out_datadriven_orig = y
                    _, out_datadriven_orig, _, out_physics_orig = self.main_model_inference(dd_model, p_model, x_adv)
                    l2_physics = self.UQ_model.myloss(out_physics_orig[out_UQ_dp<0,...].view(opt_p_num,-1), 
                                            out_datadriven_orig[out_UQ_dp<0,...].view(opt_p_num,-1))
                    p_model.optimizer.zero_grad()
                    l2_physics.backward()
                    p_model.optimizer.step()
                    train_l2_dual_p += l2_physics.item()
                    # print("batch_index",generation_stop_index, "epoch", dual_opt_index, "opt_p_l2_loss:", l2_physics.item()/opt_p_num)
            opt_dd_sum += opt_dd_num
            opt_p_sum += opt_p_num
            
        assert ((opt_p_sum!=0)or(opt_dd_sum!=0))
        if opt_p_sum == 0:
            return train_l2_dual_dd/opt_dd_sum, train_l2_dual_p, opt_dd_sum, opt_p_sum
        if opt_dd_sum == 0:
            return train_l2_dual_dd, train_l2_dual_p/opt_p_sum, opt_dd_sum, opt_p_sum
        if (opt_p_sum != 0) and (opt_dd_sum != 0):
            return train_l2_dual_dd/opt_dd_sum, train_l2_dual_p/opt_p_sum, opt_dd_sum, opt_p_sum
        
    def data_generator(self, x, y, dd_model, p_model):
        if self.modelConfig['adv_generate']:
            y = self.UQ_model.data_dict['y_normalizer_cuda'].decode(y.cuda())
            # repeat many times of gradient ascent to generate poor data
            dd_model.model.eval()
            x_adv = x.cuda()
            if self.modelConfig['adv_num'] > 0:    
                for _ in range(self.modelConfig['adv_num']):
                    x_adv.requires_grad=True
                    out = dd_model.model(x_adv).view(y.shape[0], self.S, self.S, self.T)
                    out = self.UQ_model.data_dict['y_normalizer_cuda'].decode(out)
                    l2 = self.UQ_model.myloss(out, y)
                    x_grad = torch.autograd.grad(l2, x_adv, only_inputs=True)[0].detach()
                    # gradient ascent to generate poor data
                    # x_adv = ((x_adv + 1000*(x_adv[x_adv!=0]).abs().min().item() * x_grad.sign()).clone().detach())
                    x_adv = (x_adv + self.modelConfig['epsilon'] * x_grad.sign()).clone().detach()
                x_adv.requires_grad=False
                # # 由于adv出来的数据可能并不符合physics,所以只在第一帧上做adv，后面都是solver解出来的
                # x_adv = (x_adv[:,:,:,0,:]).reshape(y.shape[0],self.S,self.S,1,T_in).repeat([1,1,1,T,1])
            return x_adv
        else:   
            # x shape: B, S, S, T, T_in, y shape: B, S, S, T
            x = self.UQ_model.data_dict['a_normalizer_cpu'].decode(x)
            y = self.UQ_model.data_dict['y_normalizer_cpu'].decode(y)
            
            x_adv = torch.concat([x[:,:,:,self.modelConfig['T']:], y], dim=-1)
            x_adv = self.UQ_model.data_dict['a_normalizer_cpu'].encode(x_adv).cuda()# * (1+ 0.001)
            # # y is after normalization
            return x_adv
        # return x.cuda()



