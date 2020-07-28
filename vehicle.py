import torch
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer



n = 5
m = 2

class Params():
    def __init__(self):
        self.L = 2.8
        self.h = 0.2
        self.lam = [1,1,10,10]


class VehicleControl():
    def __init__(self,params):
        self.params = params
        self.policyCreator()

    def next_state(self,x, u):
        e, dpsi, v, vdes, K, a, z =  x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], u[:, 0], u[:, 1]
        
        Kprob = torch.rand_like(K) < .95 # change on average every 4 seconds
        K_next = Kprob * K + ~Kprob * .1 * torch.randn_like(K)
        vprob = torch.rand_like(vdes) < .98 # change on average every 10 seconds
        vdes_next = vprob * vdes + ~vprob * torch.distributions.Uniform(3, 6).sample(sample_shape=vdes.shape)
        xnext = torch.stack([
            e + self.params.h * v * torch.sin(dpsi),
            dpsi + self.params.h * v * (K + z / self.params.L - K / (1 - e * K) * torch.cos(dpsi)),
            v + self.params.h * a,
            vdes_next,
            K_next
        ], dim=1)
        
        xnext[:, 0] += 1e-1*torch.randn(x.shape[0])
        xnext[:, 1] += 1e-2*torch.randn(x.shape[0])
        xnext[:, 2] += 1e-1*torch.randn(x.shape[0])
        
        return xnext

    def cost_batch(self,x, u):
        e, dpsi, v, vdes, K, a, z = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], u[:, 0], u[:, 1]

        return (v - vdes).pow(2) + self.params.lam[0] * e.pow(2) + self.params.lam[1] * dpsi.pow(2) + \
            self.params.lam[2] * a.pow(2) + self.params.lam[3] * z.pow(2)


    def policyCreator(self):
        # approximate value function
        S = cp.Parameter((4, 4))
        q = cp.Parameter(4)

        # dynamics
        fx = cp.Parameter(4)
        B = cp.Parameter((4, 2))

        K = cp.Parameter(1)

        u = cp.Variable(2)
        y = cp.Variable(4)

        a = u[0]
        z = u[1]

        objective = self.params.lam[2] * cp.square(a) + self.params.lam[3] * cp.square(z) + \
            cp.sum_squares(S @ y) + q @ y
        constraints = [y == fx + B @ u, cp.abs(a) <= 2, cp.abs(z + self.params.L * K) <= .68]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        self.policy = CvxpyLayer(prob, [S, q, fx, B, K], [u])


    def loss(self,time_horizon, batch_size, S, q, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        X, U = [], []
        x = torch.zeros(batch_size, n)
        x[:, 0] = .5
        x[:, 1] = .1
        x[:, 2] = 3
        x[:, 3] = 4.5
        loss = 0.0
        for _ in range(time_horizon):
            e, dpsi, v, vdes, K = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
            enext = e + self.params.h * v * torch.sin(dpsi)
            dpsi_next = dpsi + self.params.h * v * (K - K / (1 - e * K) * torch.cos(dpsi))
            fx = torch.stack([
                    enext,
                    dpsi_next,
                    v - .98*vdes-.02*4.5,
                    enext + self.params.h * v * torch.sin(dpsi_next)
                ], dim=1)
            B = torch.zeros(batch_size, 4, 2)
            B[:, 1, 1] = self.params.h * v / self.params.L
            B[:, 2, 0] = self.params.h
            B[:, 3, 1] = self.params.h * self.params.h * v * v / self.params.L
            u, = self.policy(S, q, fx, B, K.unsqueeze(-1), solver_args={"acceleration_lookback": 0})
            loss += self.cost_batch(x, u).mean() / time_horizon
            x = self.next_state(x, u)
        
        return loss

    def optimize(self,num_epochs,time_horizon = 100, batch_size=1):

        S = torch.eye(4)
        S.requires_grad_(True)
        q = torch.zeros(4, requires_grad=True)

        params = [S, q]
        opt = torch.optim.SGD(params, lr=.1)
        losses = []
        for k in range(num_epochs):
            with torch.no_grad():
                test_loss = self.loss(time_horizon, 4, S.detach(), q.detach(), seed=0)
                losses.append(test_loss)
                print("it: %03d, loss: %3.3f" % (k+1, test_loss.item()))

            opt.zero_grad()
            l = self.loss(time_horizon,batch_size, S, q, seed=k+1)
            plt.show()
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 10)
            opt.step()

        return losses,S,q
        


params = Params()

vehicle = VehicleControl(params)



losses_monte_carlo_untrained = [vehicle.loss(100, 1, torch.eye(4), torch.zeros(4), seed=1000+k) for k in range(100)]

losses,S,q = vehicle.optimize(100)


losses_monte_carlo_trained = [loss(100, 1, S.detach(), q.detach(), seed=1000+k) for k in range(100)]


print(losses_monte_carlo_untrained)

print()
print()

print(losses_monte_carlo_trained)
