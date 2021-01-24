import torch
from torch.autograd import Variable
from tqdm import tqdm


# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

def eval_model(model, test_loader, loss_func):
    correct = 0.0
    total = 0.0
    eval_loss = 0.0
    with tqdm(total=len(test_loader)) as pbar:
        for images, labels in test_loader:
            images = get_variable(images)
            labels = get_variable(labels)
            with torch.no_grad():
                outputs = model(images)
            eval_loss += loss_func(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_description("Validation: ")
            pbar.update(1)

    test_acc = 100.0 * correct / total
    eval_loss = eval_loss / total
#     print('Accuracy of the network on the test images: %d %%' % (
#         test_acc))
    return test_acc, eval_loss    

def pred_model(model, pred_loader, target, save_dir, checkname):    
    total_pred = []
    with tqdm(total=len(pred_loader)) as pbar:
        for idx, (images, _) in enumerate(pred_loader):       
            images = get_variable(images)
            
            model.eval()
            with torch.no_grad():
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.tolist()
            total_pred.extend(predicted)
            pbar.set_description("Predict Target: ")
            pbar.update(1)
    target['Label'] = total_pred
    target.to_csv('{}\{}_pred.csv'.format(save_dir, checkname), index=False)