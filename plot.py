import matplotlib.pyplot as plt
def ploter(global_acc_list,global_loss_list):
    
    plt.figure(figsize=(6,3))
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Epoch") 
    plt.plot(list(range(0,len(global_loss_list))), global_loss_list,color='red')

    plt.figure(figsize=(6,3))  
    plt.plot(list(range(0,len(global_acc_list))), global_acc_list,color='blue')
    plt.ylabel("Accuracy of Global Model")
    plt.xlabel("Epoch")  