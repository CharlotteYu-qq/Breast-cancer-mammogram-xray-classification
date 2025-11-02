import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model training options')

    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                                'efficientnet_b6', 'efficientnet_b7'])

    parser.add_argument('--csv_dir', type=str, default='CSVs')
    parser.add_argument('--batch_size', type=int, default=16,
                        choices=[16, 32, 48, 64])

    parser.add_argument('--lr', type=float, default=3e-4)
    
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='L2 regularization strength to prevent overfitting')
    
    parser.add_argument('--patience', type=int, default=25, 
                   help='Early stopping patience based on balanced accuracy')

    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--out_dir', type=str, default='breast_session')

    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    args = parser.parse_args()
    return args



# import argparse

# def get_args():
#     parser = argparse.ArgumentParser(description='Model training options')

#     parser.add_argument('--backbone', type=str, default='resnet50',
#                         choices=['resnet18', 'resnet34', 'resnet50'])

#     parser.add_argument('--csv_dir', type=str, default='CSVs')
#     parser.add_argument('--batch_size', type=int, default=48,
#                         choices=[16, 32, 48, 64])

#     parser.add_argument('--lr', type=float, default=5e-5)
    
#     parser.add_argument('--weight_decay', type=float, default=2e-4, 
#                         help='L2 regularization strength to prevent overfitting')

#     parser.add_argument('--epochs', type=int, default=35)

#     parser.add_argument('--out_dir', type=str, default='breast_session')

#     parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

#     args = parser.parse_args()
#     return args