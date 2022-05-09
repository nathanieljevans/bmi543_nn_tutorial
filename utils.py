import numpy as np 
from matplotlib import pyplot as plt 
import torch 
from celluloid import Camera
from IPython.display import HTML # to show the animation in Jupyter
import time 

def imshow2(img, ax, unnorm=False):
    if unnorm: img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    
class NN(torch.nn.Module): 
    def __init__(self, channels):
        '''
        initialize our model and paramters
        ''' 
        super().__init__()

        self.lin1 = torch.nn.Linear(2,channels) 
        self.lin2 = torch.nn.Linear(channels,channels)
        self.out = torch.nn.Linear(channels, 1)
        self.nonlin = torch.nn.ReLU()

    def forward(self, x): 
        '''
        forward pass 
        '''
        h1 = self.lin1(x)
        z1 = self.nonlin(h1)
        h2 = self.lin2(z1)
        z2 = self.nonlin(h2)
        logit = self.out(z2)
        yhat = torch.sigmoid(logit)
        return yhat

def train_model(x_train, y_train, x_test, y_test, optim=torch.optim.Adam, n_epochs=500, learning_rate=1e-2, channels=50, verbose=False):
    # init a model
    model = NN(channels=channels)

    # define our critiria (Binary Cross Entropy)
    criteria = torch.nn.BCELoss()

    # define our optimizer 
    optim = optim(model.parameters(), lr=learning_rate)

    # training loop 
    losses_train = []
    losses_test = []
    for epoch in range(n_epochs): 
        if verbose: print(f'training model... {(epoch+1)/n_epochs*100:.1f}%', end='\r')

        # make sure gradients are zero 
        model.zero_grad()

        # forward pass 
        yhat = model(x_train)

        # calculate loss of the forward pass 
        loss = criteria(yhat.squeeze(), y_train)

        # calculate parameter specific gradients 
        loss.backward() 

        # update parameter weights 
        optim.step()
        
        losses_train.append(loss.detach().item())
        with torch.no_grad(): 
            losses_test.append(criteria(model(x_test).squeeze(), y_test))
            
    return model, losses_train, losses_test



def plot_decision_boundary(model, X, y, ax):
    '''
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    inp = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float)
    out = (model(inp).view(xx.shape).detach().numpy() > 0.5)*1.
    # Plot the contour and training examples
    ax.contourf(xx, yy, out, cmap=plt.cm.Spectral)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)
    
    
class myanimator(): 
    def __init__(self): 
        # for decision boundary gif 
        self.fig, self.axes = plt.subplots(1,3, figsize=(15,5))
        self.camera = Camera(self.fig)# the camera gets the fig we'll plot
        self.first = True
        
        self.train_acc = [] 
        self.test_acc = [] 
        
        self.train_losses = [] 
        self.test_losses = [] 
        
    def add_frame(self, model, x_train, y_train, x_test, y_test, criteria): 
        
        with torch.no_grad(): 
               
            yhat_train = model(x_train).squeeze()
            loss_train = criteria(yhat_train, y_train) 
            self.train_acc.append(((yhat_train > 0.5) == y_train).float().mean())
            self.train_losses.append(loss_train.detach().numpy().item())
            
            
            yhat_test = model(x_test).squeeze()
            loss_test = criteria(yhat_test, y_test)
            self.test_acc.append(((yhat_test > 0.5) == y_test).float().mean())
            self.test_losses.append(loss_test)
        
        
            plot_decision_boundary(model, x_train, y_train, self.axes[0])
            plot_decision_boundary(model, x_test, y_test, self.axes[1])
            
            if self.first: 
                self.axes[0].set_title('train data')
                self.axes[1].set_title('test data')
                self.axes[2].set_title('performance')
                self.axes[2].set_xlabel('epoch')
                self.axes[2].plot(self.train_acc, 'r-', label='train acc.')
                self.axes[2].plot(self.train_losses, 'b-', label='train loss')
                self.axes[2].plot(self.test_acc, 'r--', label='test acc.')
                self.axes[2].plot(self.test_losses, 'b--', label='test loss')
                self.axes[2].legend()
                self.first = False
            else: 
                self.axes[2].plot(self.train_acc, 'r-')
                self.axes[2].plot(self.train_losses, 'b-')
                self.axes[2].plot(self.test_acc, 'r--')
                self.axes[2].plot(self.test_losses, 'b--')
            self.camera.snap()
            
    def show(self):
        tic = time.time()
        print()
        print('generating animation...')#, end='')
        animation = self.camera.animate() # animation ready
        gif = HTML(animation.to_html5_video()) # displaying the animation
        plt.close('all')
        plt.gcf()
        #print(f"duration: {(time.time() - tic).2f} s")
        return gif