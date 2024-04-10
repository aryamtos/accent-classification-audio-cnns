import torch
import torch.nn as nn
from sklearn.model_selection import KFold



def cross_validation_(k_folds,n_epochs,loss,dataset,
                        optimizer,model,criterion,loss_list:list,device,results,acc_list:list):

    torch.manual_seed(42)

    kfold = KFold(n_splits=k_folds,shuffle=True)

    
    for fold, (train_ids,test_ids) in enumerate(kfold.split(dataset)):

        print(f'FOLD {fold}')
        print('--------------------------------')
    
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=10, 
                        sampler=train_subsampler)

        testloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=10,
                    sampler= test_subsampler)
        
        #criterion =nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(0, n_epochs):

            print(f'Starting epoch {epoch+1}')

            current_loss = 0.0

            for i, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, targets)

                loss.backward()

                optimizer.step()

                current_loss += loss.item()
                
                if i % 50 == 49:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 50))
                    loss_list.append(current_loss)
                    current_loss = 0.0

        print('Training process has finished. Saving trained model.')

        print('Starting testing')

        save_path = f'./model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)

        correct, total = 0, 0
        #with torch.no_grad():
        for i, data in enumerate(testloader, 0):

            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)
        acc_list.append(results[fold])

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

#torch.save(model.state_dict(), '/content/pesos.pth')









