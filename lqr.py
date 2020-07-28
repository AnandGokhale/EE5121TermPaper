import torch
import cvxpy as cp
import numpy as np

from cvxpylayers.torch import CvxpyLayer
from scipy.linalg import sqrtm



# Generate system Matrices
def gensys(N,M,noise = 0.25,seed = 0):
    np.random.seed(seed)
    A = np.random.randn(N,N)
    A/=np.max(np.abs(np.linalg.eig(A)[0]))
    B = np.random.randn(N,M)
    W = noise * np.eye(N)

    return A,B,W


# Ideal Value
def lqr(A,B,Q,R,W):
    N,M = B.shape

    P = cp.Variable((N, N), PSD=True)

    objective = cp.trace(P@W)
    constraints = [cp.bmat([
        [R + B.T@P@B, B.T@P@A],
        [A.T@P@B, Q+A.T@P@A-P]
    ]) >> 0, P >> 0]
    result = cp.Problem(cp.Maximize(objective), constraints).solve()

    return P.value

N = 4
M = 2

A,B,W = gensys(N,M)



class COCP():
    def __init__(self,A,B,W,Q,R,time_horizon,batch_size):
        self.A  = A
        self.B  = B
        self.Q  = Q
        self.R  = R

        self.time = time_horizon
        self.batch_size = batch_size

        self.At, self.Bt, self.Qt, self.Rt = map(torch.from_numpy, [A, B, Q, R])
        self.Q_batch = self.Qt.repeat(batch_size, 1, 1)
        self.R_batch = self.Rt.repeat(batch_size, 1, 1)
        self.A_batch = self.At.repeat(batch_size, 1, 1)
        self.B_batch = self.Bt.repeat(batch_size, 1, 1)


        self.policyCreator()


    def policyCreator(self):
        x = cp.Parameter((N,1))
        P_sqrt = cp.Parameter((N,N))

        u = cp.Variable((M,1))
        x_next = cp.Variable((N,1))

        objective = cp.quad_form(u, self.R) + cp.sum_squares(P_sqrt @ x_next)
        constraints = [x_next == self.A @ x + self.B @ u]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        self.policy = CvxpyLayer(prob, [x, P_sqrt], [u])

    def next_state(self,x,u):
        return  torch.bmm(self.A_batch, x) + \
                torch.bmm(self.B_batch, u) + \
                0.25**.5 * torch.randn(self.batch_size, N, 1).double()        

    def loss(self, P_sqrt, seed  = None):
        if seed is not None:
            torch.manual_seed(seed)
        x_batch = torch.randn(self.batch_size, N, 1).double()
        P_sqrt_batch = P_sqrt.repeat(self.batch_size, 1, 1)

        loss = 0.0
        for _ in range(self.time):
            u_batch, = self.policy(x_batch, P_sqrt_batch, solver_args={"acceleration_lookback": 0})
            state_cost = torch.bmm(torch.bmm(self.Q_batch, x_batch).transpose(2, 1), x_batch)
            control_cost = torch.bmm(torch.bmm(self.R_batch, u_batch).transpose(2, 1), u_batch)
            cost_batch = (state_cost.squeeze() + control_cost.squeeze())
            loss += cost_batch.sum() / (self.time * self.batch_size)
            x_batch = self.next_state(x_batch,u_batch)
            
        return loss

    def optimize(self,num_epochs = 50,P_sqrt_init  = None):
        if(P_sqrt_init is None):
            P_sqrt = torch.eye(N).double()
        else:
            P_sqrt = P_sqrt_init    
        P_sqrt.requires_grad_(True)
        losses = []
        opt = torch.optim.SGD([P_sqrt], lr = 0.5)
        for k in range(num_epochs):
            with torch.no_grad():
                test_loss = self.loss(P_sqrt.detach(), seed=0).item()
                K_np = (torch.solve(-self.Bt.t() @ P_sqrt.t() @ P_sqrt @ self.At, self.Rt + self.Bt.t() @ P_sqrt.t() @ P_sqrt @ self.Bt).solution).detach().numpy()
                dist = np.linalg.norm(K_np - Kt)
                P = (P_sqrt.t() @ P_sqrt).detach().numpy()
                dist_P = np.linalg.norm(P_lqr - P)
                losses.append(test_loss)
                print("it: %03d, loss: %3.3f, dist: %3.3f, dist_P: %3.3f" % (k+1, test_loss - loss_lqr, dist, dist_P))
            opt.zero_grad()
            l = self.loss(P_sqrt, seed=k+1)
            l.backward()
            opt.step()
            if l<0.01:
                opt = torch.optim.SGD([P_sqrt], lr=.1)

        return losses
        
P_lqr = lqr(A,B,np.eye(N),np.eye(M),W)

LQR = COCP(A,B,W,np.eye(N),np.eye(M),100,20)

R = np.eye(M)
Q = np.eye(N)

import matplotlib
import matplotlib.pyplot as plt

Kt = np.linalg.solve(R + B.T @ P_lqr @ B, -B.T @ P_lqr @ A)
loss_lqr = LQR.loss(torch.from_numpy(sqrtm(P_lqr)), seed=0).item()
print("Loss LQR : ",loss_lqr)

losses = LQR.optimize()

plt.semilogy(losses, color='k', label='COCP')
plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
plt.axhline(loss_lqr, linestyle='--', color='k', label='LQR')
plt.ylabel("cost")
plt.xlabel("Iterations")
plt.subplots_adjust(left=.15, bottom=.2)
plt.savefig("lqr.pdf")
plt.show()