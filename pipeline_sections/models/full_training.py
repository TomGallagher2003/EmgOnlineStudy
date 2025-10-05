import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import psutil, GPUtil, time, random
import pynvml
# Set random seeds for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Dataset Class ---
class EMGDataset(Dataset):
    def __init__(self, data, labels):
        """
        Initialize EMG Dataset.
        
        Args:
            data (numpy.ndarray): EMG data array
            labels (numpy.ndarray): Corresponding labels
        """
        self.data = data
        self.labels = labels
    
    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (sample, label) where sample is a tensor of EMG data and label is its corresponding class
        """
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --- Model Architecture ---
class CNN1D(nn.Module):
    def __init__(self, input_channels, length, embed_dim):
        """
        1D CNN for feature extraction from EMG signals.
        
        Args:
            input_channels (int): Number of input channels (EMG sensors)
            length (int): Length of the input signal
            embed_dim (int): Embedding dimension for transformer
        """
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        """Forward pass through the CNN layers."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes):
        """
        Transformer model for classification.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            num_classes (int): Number of output classes
        """
        super(TransformerModel, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(128 * embed_dim, num_classes)
    
    def forward(self, x):
        """Forward pass through the transformer."""
        x = self.transformer(x)  # Transformer encoder
        x = x.reshape(-1,x.shape[1]*x.shape[2])  # Flatten
        x = self.fc(x)  # Output classification results
        return x

class CNN1D_Transformer(nn.Module):
    def __init__(self, input_channels, length, embed_dim, num_heads, num_layers, num_classes):
        """
        Combined CNN-Transformer model for EMG classification.
        
        Args:
            input_channels (int): Number of input channels
            length (int): Length of input signal
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            num_classes (int): Number of output classes
        """
        super(CNN1D_Transformer, self).__init__()
        self.cnn1d = CNN1D(input_channels, length, embed_dim)
        self.transformer = TransformerModel(length//4, num_heads, num_layers, num_classes)
    
    def forward(self, x):
        """Forward pass through CNN followed by transformer."""
        x = self.cnn1d(x)
        x = self.transformer(x)
        return x

def process_emg_data(file_path, sample_size=512):
    """
    Process EMG data and segment it based on labels.
    
    Args:
        file_path (str): Path to HDF5 file
        sample_size (int): Size of each sample segment, default 512
        
    Returns:
        tuple: (sampled_segments, sampled_labels) where:
            sampled_segments (numpy.ndarray): Processed EMG data with shape (n, sample_size, 300)
            sampled_labels (numpy.ndarray): Corresponding labels with shape (n,)
    """
    
    # Open HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Get data and labels
        emg_data = file['/emg'][:]
        restimulus_labels = file['/restimulus'][:]

    # Ensure emg_data and restimulus_labels have same length
    assert len(emg_data) == len(restimulus_labels), "EMG data and labels must have the same length."
    
    # Lists to store segmented data and labels
    segments = []
    labels = []

    # Initialize start index for current segment
    start_idx = 0

    # Iterate through restimulus_labels to find label change points
    for i in range(1, len(restimulus_labels)):
        if restimulus_labels[i] != restimulus_labels[i - 1]:
            # Extract current segment data and label
            segment_data = emg_data[start_idx:i]  # Get data from start_idx to i
            segment_label = restimulus_labels[start_idx]  # Current segment label
            
            # Save data and label
            segments.append(segment_data)
            labels.append(segment_label)
            
            # Update segment start index
            start_idx = i

    # Process last segment (from last label change to end)
    segment_data = emg_data[start_idx:]  # Get last segment data
    segment_label = restimulus_labels[start_idx]  # Current segment label
    segments.append(segment_data)
    labels.append(segment_label)
    
    # Sample segments using sliding window
    sampled_segments = []
    sampled_labels = []
    window_step = 512
    
    for i, segment_data in enumerate(segments):
        label = labels[i]
        segment_length = len(segment_data)
        
        # If segment length >= sample size, extract samples using sliding window
        if segment_length >= sample_size:
            start_index = 0
            while start_index + sample_size <= segment_length:
                end_index = start_index + sample_size
                sampled_segment = segment_data[start_index:end_index]
                sampled_segment = np.expand_dims(sampled_segment, axis=0)
                
                # Save sampled data and label
                sampled_segments.append(sampled_segment)
                sampled_labels.append(label)
                
                # Move window by step size
                start_index += window_step

    # Concatenate all sampled segments along first dimension: (n, 1, sample_size, 300) -> (n, sample_size, 300)
    sampled_segments = np.concatenate(sampled_segments, axis=0)
    sampled_labels = np.array(sampled_labels)  # Convert to numpy array
    sampled_segments = normalize_per_movement(sampled_segments, sampled_labels)  # Normalize per movement
    return sampled_segments, sampled_labels

def remove_movement(data: np.ndarray, labels: np.ndarray, movement: list[int]):
    """
    Remove specified movement labels and corresponding data.
    
    Args:
        data (np.ndarray): Data array with shape (num_samples, ...)
        labels (np.ndarray): Label array with shape (num_samples,)
        movement (list[int]): List of movement labels to remove
        
    Returns:
        tuple: (new_data, new_labels) filtered arrays
    """
    mask = ~np.isin(labels, movement)  # Find indices not in movement list
    new_data = data[mask]
    new_labels = labels[mask]
    
    # Print class distribution
    unique_labels, counts = np.unique(new_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: #Sample: {count}")

    print(f"Data shape: {new_data.shape}, Label shape: {new_labels.shape}")
    
    return new_data, new_labels

def normalize_per_movement(data, labels):
    """Normalize data separately for each movement class."""
    unique_labels = np.unique(labels)
    scaler = MinMaxScaler()
    
    for label in unique_labels:
        # Get indices for current class
        indices = np.where(labels == label)[0]
        # Extract class data
        class_data = data[indices]
        # Normalize class data
        class_data_normalized = scaler.fit_transform(class_data.reshape(-1, class_data.shape[-1])).reshape(class_data.shape)
        # Update data
        data[indices] = class_data_normalized
    
    return data

def process_h5_files(folder_path, sample_size=512, max_zero_samples=4000):
    """Process HDF5 files to extract and normalize EMG data."""
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    data, labels = [], []

    for file_name in h5_files:
        file_path = os.path.join(folder_path, file_name)
        sampled_segments, sampled_labels = process_emg_data(file_path, sample_size=sample_size)
        data.append(sampled_segments)
        labels.append(sampled_labels)

    # Concatenate data from all files
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Handle label 0 samples (rest state)
    zero_label_indices = np.where(labels == 0)[0]

    # If more than max_zero_samples label 0 samples, randomly select max_zero_samples
    if len(zero_label_indices) > max_zero_samples:
        selected_indices = np.random.choice(zero_label_indices, size=max_zero_samples, replace=False)
    else:
        selected_indices = zero_label_indices  # Keep all if less than max_zero_samples

    # Get indices for non-zero labels
    non_zero_label_indices = np.where(labels != 0)[0]

    # Combine selected indices
    final_indices = np.concatenate([selected_indices, non_zero_label_indices])
    data=data[final_indices]
    labels=labels[final_indices]
    
    # Print class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: #Sample: {count}")

    print(f"Data shape: {data.shape}, Label shape: {labels.shape}")
    return data,labels 

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, saving_path, i):
    """Train the model and evaluate on validation set."""
    model.train()
    best_accuracy = 0.0  # Initialize best accuracy
    results = []
    
    # Initialize GPU monitoring
    gpu = None
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get first GPU
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Track resources before epoch
        cpu_ram_before = psutil.virtual_memory().used / (1024 ** 3)  # GB
        if gpu:
            vram_before = gpu.memoryUsed  # MB
        
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Track resources after epoch
        cpu_ram_after = psutil.virtual_memory().used / (1024 ** 3)  # GB
        if gpu:
            vram_after = gpu.memoryUsed  # MB
            vram_usage = vram_after - vram_before
        else:
            vram_usage = 0
            
        epoch_time = time.time() - epoch_start_time
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        # Validation
        val_accuracy, val_preds, val_labels = evaluate_model(model, val_loader, device)
        
        # Collect results
        epoch_result = {
            'Epoch': epoch + 1,
            'Training Loss': epoch_loss,
            'Training Acc': epoch_accuracy,
            'Val Acc': val_accuracy,
            'Epoch Time (s)': epoch_time,
            'CPU RAM Usage (GB)': cpu_ram_after - cpu_ram_before,
            'VRAM Usage (MB)': vram_usage
        }
        results.append(epoch_result)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {epoch_loss:.4f}, "
              f"Train Acc: {epoch_accuracy:.2f}%, "
              f"Val Acc: {val_accuracy:.2f}%, "
              f"Time: {epoch_time:.2f}s, "
              f"RAM Δ: {cpu_ram_after - cpu_ram_before:.2f}GB, "
              f"VRAM Δ: {vram_usage:.2f}MB")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), saving_path + f'/best_model_weight_{best_accuracy:.2f}_{i+1}_{timestamp}.pth')
            torch.save(model, saving_path + f'/best_model_all_{best_accuracy:.2f}_{i+1}_{timestamp}.pth')
            print(f"New best accuracy: {best_accuracy:.2f}%, model saved.")

    return best_accuracy, results, timestamp

def evaluate_model(model, data_loader, device):
    """Evaluate model and return accuracy, predictions, and labels."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Evaluate Acc:{accuracy:.2f}")
    return accuracy, all_preds, all_labels

def save_results_to_excel(model, input_shape, results, output_file, hardware_stats):
    """Save model summary, hardware stats, and training results to Excel file."""
    # Create model summary
    model_summary = get_model_summary(model, input_shape)
    
    # Create results DataFrame
    results_dfs = []
    for i, trial_results in enumerate(results):
        trial_df = pd.DataFrame(trial_results)
        trial_df['Trial'] = i + 1
        results_dfs.append(trial_df)
    
    all_results_df = pd.concat(results_dfs)
    
    # Create hardware stats DataFrame
    hardware_df = pd.DataFrame(hardware_stats)
    
    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 1. Model Summary sheet
        model_summary.to_excel(writer, sheet_name='Model Summary', index=False)
        
        # 2. Hardware Monitoring sheet
        hardware_df.to_excel(writer, sheet_name='Hardware Monitor', index=False)
        
        # 3. Group by trial and save each trial to separate sheet
        for trial_num, group in all_results_df.groupby('Trial'):
            group.drop('Trial', axis=1).to_excel(
                writer, 
                sheet_name=f'Trial_{trial_num}', 
                index=False
            )
        
        # 4. Save all results to a summary sheet
        all_results_df.to_excel(
            writer, 
            sheet_name='All Trials', 
            index=False
        )

def get_model_summary(model, input_shape):
    """Extract model architecture details (layers, parameters, output shapes)."""
    model_summary = {
        "Layer Name": [],
        "Layer Type": [],
        "Output Shape": [],
        "Param #": [],
        "Trainable": []
    }

    def register_hook(module):
        def hook(module, input, output):
            layer_name = str(module.__class__.__name__)
            model_summary["Layer Name"].append(layer_name)
            model_summary["Layer Type"].append(str(module))
            model_summary["Param #"].append(sum(p.numel() for p in module.parameters()))
            model_summary["Trainable"].append(any(p.requires_grad for p in module.parameters()))
            
            # Handle output shape carefully
            if output is None:
                model_summary["Output Shape"].append("None")
            elif isinstance(output, (tuple, list)):
                try:
                    shapes = [list(o.shape) if o is not None else "None" for o in output]
                    model_summary["Output Shape"].append(shapes)
                except AttributeError:
                    model_summary["Output Shape"].append("N/A")
            else:
                try:
                    model_summary["Output Shape"].append(list(output.shape))
                except AttributeError:
                    model_summary["Output Shape"].append("N/A")

        # Only register hook if module is not Sequential/ModuleList and has parameters
        if (not isinstance(module, nn.Sequential) and 
            not isinstance(module, nn.ModuleList) and 
            len(list(module.children())) == 0):  # Only leaf modules
            return module.register_forward_hook(hook)
        return None

    hooks = []
    # Register hooks on all leaf modules
    for module in model.modules():
        hook_handle = register_hook(module)
        if hook_handle is not None:
            hooks.append(hook_handle)
    
    # Forward pass with dummy input
    try:
        dummy_input = torch.zeros(*input_shape).to(next(model.parameters()).device)
        model(dummy_input)
    except Exception as e:
        print(f"Error during forward pass: {e}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

    return pd.DataFrame(model_summary)

def get_system_stats(stage=""):
    """获取系统资源使用情况"""
    stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
    }
    
    # CPU RAM
    ram = psutil.virtual_memory()
    stats["cpu_ram_total_gb"] = ram.total / (1024 ** 3)
    stats["cpu_ram_used_gb"] = ram.used / (1024 ** 3)
    stats["cpu_ram_percent"] = ram.percent
    
    # GPU VRAM
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        vram_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        stats["gpu_vram_total_gb"] = vram_info.total / (1024 ** 3)
        stats["gpu_vram_used_gb"] = vram_info.used / (1024 ** 3)
        stats["gpu_vram_percent"] = (vram_info.used / vram_info.total) * 100
        pynvml.nvmlShutdown()
    except:
        stats["gpu_vram_total_gb"] = 0
        stats["gpu_vram_used_gb"] = 0
        stats["gpu_vram_percent"] = 0
    
    return stats

def record_hardware_stats(file_path, stats):
    """记录硬件状态到Excel"""
    try:
        # 尝试读取现有Excel文件
        df = pd.read_excel(file_path, sheet_name="hardware_monitor")
    except:
        # 如果文件不存在，创建新的DataFrame
        df = pd.DataFrame(columns=stats.keys())
    
    # 添加新数据
    new_row = pd.DataFrame([stats])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # 保存回Excel
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="hardware_monitor", index=False)

if __name__ == "__main__":
    all_results = []
    hardware_stats = []  # To store all hardware monitoring data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saving_path = 'IMECE2025_results/full_training'
    os.makedirs(saving_path, exist_ok=True)
    output_file = os.path.join(saving_path, f'training_results_{timestamp}.xlsx')
    
    # Record initial state
    initial_stats = get_system_stats("initial_state")
    hardware_stats.append(initial_stats)
    
    # Training Loop (5 trials)
    for i in range(10):
        print(f"\n=== Trial {i + 1} ===")
        
        # Record trial start state
        trial_start_stats = get_system_stats(f"trial_{i+1}_start")
        hardware_stats.append(trial_start_stats)
        
        
        # Hyperparameters
        input_channels = 8                      # EMG channels
        sample_size = 512                       # Window size
        embed_dim = 128                         # Transformer embedding dimension
        num_heads = 8                           # Transformer attention heads
        num_layers = 3                          # Transformer layers
        num_classes = 18                        # Movement classes
        learning_rate = 0.001
        num_epochs = 30
        batch_size = 512
        
        input_shape = (batch_size, input_channels, sample_size) # for saving the model information
        length = sample_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        folder_path1 = r"D:\Data\Ninapro_dataset\MIXED\hdf5_format\EB_norm_filtered"
        
        # Record pre-data loading state
        pre_data_stats = get_system_stats(f"trial_{i+1}_pre_data_loading")
        hardware_stats.append(pre_data_stats)
        
        # Data Loading
        data1, labels1 = process_h5_files(folder_path1, sample_size=sample_size, max_zero_samples=6000) 
        data = np.concatenate([data1], axis=0)
        labels = np.concatenate([labels1], axis=0)
        
        # Record post-data loading state
        post_data_stats = get_system_stats(f"trial_{i+1}_post_data_loading")
        hardware_stats.append(post_data_stats)
        
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42) 
        print(f"Train data shape: {X_train.shape}, Train label shape: {y_train.shape}")
        print(f"Val data shape: {X_val.shape}, Val label shape: {y_val.shape}")
        print(f"Device: {device}")
        train_dataset = EMGDataset(X_train, y_train)
        val_dataset = EMGDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        # Record pre-model creation state
        pre_model_stats = get_system_stats(f"trial_{i+1}_pre_model_creation")
        hardware_stats.append(pre_model_stats)
        
        #create model and initial loss function and optimizer
        model = CNN1D_Transformer(input_channels, length, embed_dim, num_heads, num_layers, num_classes).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Record pre-training state
        pre_train_stats = get_system_stats(f"trial_{i+1}_pre_training")
        hardware_stats.append(pre_train_stats)
        
        # training model
        best_acc, results, trial_timestamp = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, saving_path, i)
        all_results.append(results)
        
        # Record post-training state
        post_train_stats = get_system_stats(f"trial_{i+1}_post_training")
        hardware_stats.append(post_train_stats)
        
        
        # test model
        model = torch.load(saving_path + '/best_model_all_{:.2f}_{}_{}.pth'.format(best_acc, i + 1, trial_timestamp))
        model.to(device)  # Ensure model is on correct device (CPU or GPU)
        Eval_acc, preds, labels = evaluate_model(model, val_loader, device)
        
        report = classification_report(labels, preds, digits=4)

        with open(saving_path + "/classification_report_{:.2f}_{}_{}.txt".format(best_acc, i + 1, trial_timestamp), "w") as f:
            f.write(f"Total parameters: {total_params}\n\n")
            f.write(report)
        
        # Plot confusion matrix
        conf_mat = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 7))
        num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=num, yticklabels=num)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(saving_path + '/confusion_matrix_full_{:.2f}_{}_{}.png'.format(best_acc, i+1, trial_timestamp))
    
    # Saving Path for the Excel file 
    output_file = saving_path + f'/training_results_{timestamp}.xlsx'
    # Save all results at the end
    save_results_to_excel(model, input_shape, all_results, output_file, hardware_stats)