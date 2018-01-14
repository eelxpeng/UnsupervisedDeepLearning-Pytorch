import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np
import math
from sklearn.mixture import GaussianMixture

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = max(init_lr * (0.9 ** (epoch//10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

log2pi = math.log(2*math.pi)
def log_likelihood_samples_unit_gaussian(samples):
    return -0.5*log2pi*samples.size()[1] - torch.sum(0.5*(samples)**2, 1)

def log_likelihood_samplesImean_sigma(samples, mu, logvar):
    return -0.5*log2pi*samples.size()[1] - torch.sum(0.5*(samples-mu)**2/torch.exp(logvar) + 0.5*logvar, 1)

class VaDE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_centroids=10, binary=True,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500]):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.n_centroids = n_centroids
        self.encoder = buildNetwork([input_dim] + encodeLayer)
        self.decoder = buildNetwork([z_dim] + decodeLayer)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

        self.create_gmmparam(n_centroids, z_dim)

    def create_gmmparam(self, n_centroids, z_dim):
        self.theta_p = nn.Parameter(torch.ones(n_centroids)/n_centroids)
        self.u_p = nn.Parameter(torch.zeros(z_dim, n_centroids))
        self.lambda_p = nn.Parameter(torch.ones(z_dim, n_centroids))

    def initialize_gmm(self, dataloader):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        self.eval()
        data = []
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs, mu, logvar = self.forward(inputs)
            data.append(z.data.cpu().numpy())
        data = np.concatenate(data)
        gmm = GaussianMixture(n_components=self.n_centroids,covariance_type='diag')
        gmm.fit(data)
        self.u_p.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.lambda_p.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          # num = np.array([[ 1.096506  ,  0.3686553 , -0.43172026,  1.27677995,  1.26733758,
          #       1.30626082,  0.14179629,  0.58619505, -0.76423112,  2.67965817]], dtype=np.float32)
          # num = np.repeat(num, mu.size()[0], axis=0)
          # eps = Variable(torch.from_numpy(num))
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def get_gamma(self, z, z_mean, z_log_var):
        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
        z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK

        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
            (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma

    def loss_function(self, recon_x, x, z, z_mean, z_log_var):
        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
        z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
        u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
        lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
        theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK
        
        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
            (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK
        
        BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
            (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
        logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+\
            torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
        qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
        logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
        logqcx = torch.sum(torch.log(gamma)*gamma, 1)

        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE + logpzc + qentropy + logpc + logqcx)

        return loss

    #===============================================================
    # below is defined according to the released code by the authors
    # However, they are incorrect in several places
    #===============================================================

    # def get_gamma(self, z, z_mean, z_log_var):
    #     Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
    #     z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
    #     z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
    #     u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
    #     lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
    #     theta_tensor3 = self.theta_p.unsqueeze(0).unsqueeze(1).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK

    #     p_c_z = torch.exp(torch.sum(torch.log(theta_tensor3) - 0.5*torch.log(2*math.pi*lambda_tensor3)-\
    #         (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
    #     gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK

    #     return gamma

    # def loss_function(self, recon_x, x, z, z_mean, z_log_var):
    #     Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
    #     z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
    #     z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
    #     u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
    #     lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
    #     theta_tensor3 = self.theta_p.unsqueeze(0).unsqueeze(1).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK

    #     p_c_z = torch.exp(torch.sum(torch.log(theta_tensor3) - 0.5*torch.log(2*math.pi*lambda_tensor3)-\
    #         (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
    #     gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK
    #     gamma_t = gamma.unsqueeze(1).expand(gamma.size(0), self.z_dim, gamma.size(1)) #
        
    #     BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
    #         (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
    #     logpzc = torch.sum(torch.sum(0.5*gamma_t*(self.z_dim*math.log(2*math.pi)+torch.log(lambda_tensor3)+\
    #         torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3), dim=1), dim=1)
    #     qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
    #     logpc = -torch.sum(torch.log(self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_centroids))*gamma, 1)
    #     logqcx = torch.sum(torch.log(gamma)*gamma, 1)

    #     loss = torch.mean(BCE + logpzc + qentropy + logpc + logqcx)

    #     # return torch.mean(qentropy)
    #     return loss

    def forward(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, 
        visualize=False, anneal=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        # validate
        self.eval()
        valid_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs, mu, logvar = self.forward(inputs)

            loss = self.loss_function(outputs, inputs, z, mu, logvar)
            valid_loss += loss.data[0]*len(inputs)
            # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
            # total_num += inputs.size()[0]

        # valid_loss = total_loss / total_num
        print("#Epoch -1: Valid Loss: %.5f" % (valid_loss / len(validloader.dataset)))

        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            if anneal:
                epoch_lr = adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                z, outputs, mu, logvar = self.forward(inputs)
                loss = self.loss_function(outputs, inputs, z, mu, logvar)
                train_loss += loss.data[0]*len(inputs)
                loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.data[0]))

            # validate
            self.eval()
            valid_loss = 0.0
            Y = []
            Y_pred = []
            for batch_idx, (inputs, labels) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, outputs, mu, logvar = self.forward(inputs)

                loss = self.loss_function(outputs, inputs, z, mu, logvar)
                valid_loss += loss.data[0]*len(inputs)
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]
                gamma = self.get_gamma(z, mu, logvar).data.cpu().numpy()
                Y.append(labels.numpy())
                Y_pred.append(np.argmax(gamma, axis=1))

                # view reconstruct
                if visualize and batch_idx == 0:
                    n = min(inputs.size(0), 8)
                    comparison = torch.cat([inputs.view(-1, 1, 28, 28)[:n],
                                            outputs.view(-1, 1, 28, 28)[:n]])
                    save_image(comparison.data.cpu(),
                                 'results/vae/reconstruct/reconstruction_' + str(epoch) + '.png', nrow=n)

            Y = np.concatenate(Y)
            Y_pred = np.concatenate(Y_pred)
            acc = cluster_acc(Y_pred, Y)
            # valid_loss = total_loss / total_num
            print("#Epoch %3d: lr: %.5f, Train Loss: %.5f, Valid Loss: %.5f, acc: %.5f" % (
                epoch, epoch_lr, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset), acc[0]))

            # view sample
            if visualize:
                sample = Variable(torch.randn(64, self.z_dim))
                if use_cuda:
                   sample = sample.cuda()
                sample = self.decode(sample).cpu()
                save_image(sample.data.view(64, 1, 28, 28),
                           'results/vae/sample/sample_' + str(epoch) + '.png')

    def log_marginal_likelihood_estimate(self, x, num_samples):
        weight = torch.zeros(x.size(0))
        for i in range(num_samples):
            z, recon_x, mu, logvar = self.forward(x)
            zloglikelihood = log_likelihood_samples_unit_gaussian(z)
            dataloglikelihood = torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
            log_qz = log_likelihood_samplesImean_sigma(z, mu, logvar)
            weight += torch.exp(dataloglikelihood + zloglikelihood - log_qz).data
        # pdb.set_trace()
        return torch.log(torch.clamp(weight/num_samples, min=1e-40))

