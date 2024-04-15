import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class AnomalyDetector(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(AnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.sigmoid(self.fc4(out))
        return out


# Define the hyperparameters
input_size = 5
hidden_size1 = 128
hidden_size2 = 256
hidden_size3 = 128
learning_rate = 0.01
num_epochs = 100

model = AnomalyDetector(input_size, hidden_size1, hidden_size2, hidden_size3)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def get_datasets(
    train_normal_data,
    test_normal_data,
    test_anomalous_data,
    val_normal_data,
    val_anomalous_data,
):
    # Prepare data
    train_normal_data = torch.tensor(train_normal_data, dtype=torch.float32)
    test_normal_data = torch.tensor(test_normal_data, dtype=torch.float32)
    test_anomalous_data = torch.tensor(test_anomalous_data, dtype=torch.float32)
    val_normal_data = torch.tensor(val_normal_data, dtype=torch.float32)
    val_anomalous_data = torch.tensor(val_anomalous_data, dtype=torch.float32)

    # Define DataLoader
    train_loader = DataLoader(
        TensorDataset(train_normal_data), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(val_normal_data), batch_size=32)
    test_loader_normal = DataLoader(TensorDataset(test_normal_data), batch_size=32)
    test_loader_anomalous = DataLoader(
        TensorDataset(test_anomalous_data), batch_size=32
    )

    return train_loader, val_loader, test_loader_normal, test_loader_anomalous


def train_model(train_loader, val_loader):
    # Training the model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            inputs = batch_data[0]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data[0]
                outputs = model(inputs)
                print(outputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader.dataset):.4f}, Val Loss: {val_loss/len(val_loader.dataset):.4f}"
        )


def evaluate_model(loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item() * inputs.size(0)

    return test_loss / len(loader.dataset)
