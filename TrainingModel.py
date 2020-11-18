import torch

class train_model():
    def __init__(self,model,train_loader,val_loader,optimizer,criterion,epochs,N_test,path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.N_test = N_test
        self.PATH = path
        self.epochs = epochs
        self.accuracy_list = []
        self.cost_list = []
        self.loss_list = []

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch} started")
            self.model.train()
            cost = 0
            for x,y in self.train_loader:
                self.optimizer.zero_grad()
                z = self.model(x)
                loss = self.criterion(z,y)
                loss.backward()
                self.optimizer.step()
                self.loss_list.append(loss)
                cost += loss.item()
            self.cost_list.append(cost)

            correct = 0
            self.model.eval()
            for x_test, y_test in self.val_loader:
                z = self.model(x_test)
                _, yhat = torch.max(z.data,1)
                correct += (yhat == y_test).sum().item()
            accuracy = correct/self.N_test
            print(accuracy)
            self.accuracy_list.append(accuracy)
            print(f"Epoch {epoch} Ended")

        torch.save(self.model,self.PATH)

        return self.accuracy_list, self.loss_list, self.cost_list