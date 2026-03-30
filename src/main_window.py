import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QLineEdit, QFileDialog, QVBoxLayout, QWidget, 
                             QMessageBox, QComboBox, QTableWidget, QTableWidgetItem,
                             QHBoxLayout, QHeaderView)
from PyQt5.QtCore import Qt

# It's better to have other modules not create plots automatically.
# This should be controlled by the main application.
# For now, we assume they are refactored to not show plots.
from lcd_fastica import fast_ica_processing
from build_tensor import build_tensor_data
from train_model import train_model

# --- Directory Setup ---
# Get the absolute path of the directory containing this script (src)
# BASE_DIR should point to the project root, not 'src'
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, '..', 'processed_data')
ICA_RESULTS_DIR = os.path.join(PROCESSED_DATA_DIR, 'ica_results')
TENSOR_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'tensor_dataset')
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, '..', 'models')

# Ensure all directories exist
os.makedirs(ICA_RESULTS_DIR, exist_ok=True)
os.makedirs(TENSOR_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信号处理与模型训练 ")
        self.setGeometry(100, 100, 600, 700)
        
        # Define the label-to-integer mapping here
        self.label_map = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9
        }
        
        self.initUI()
    
    def initUI(self):
        # --- Top Level: File Selection ---
        self.label_file = QLabel("1. 选择NPY文件:")
        self.line_edit_file = QLineEdit()
        self.button_file = QPushButton("浏览")
        self.button_file.clicked.connect(self.browse_file)
        
        self.label_sampling_rate = QLabel("采样率(Hz):")
        self.line_edit_sampling_rate = QLineEdit("20000") # Default value
        
        self.label_label = QLabel("选择标签:")
        self.combo_label = QComboBox()
        self.combo_label.addItems(self.label_map.keys())
        
        self.button_add_to_batch = QPushButton("添加至批处理列表")
        self.button_add_to_batch.clicked.connect(self.add_to_batch)

        # --- Middle Level: Batch View ---
        self.label_batch = QLabel("2. 批处理列表:")
        self.table_batch = QTableWidget()
        self.table_batch.setColumnCount(3)
        self.table_batch.setHorizontalHeaderLabels(["NPY 文件路径", "标签", "状态"])
        self.table_batch.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_batch.setAlternatingRowColors(True)

        # --- Bottom Level: Training Setup and Execution ---
        self.label_train_setup = QLabel("3. 训练设置与执行:")
        self.label_train_mode = QLabel("选择训练模式:")
        self.combo_train_mode = QComboBox()
        self.combo_train_mode.addItems(['第一次训练模型', '读取已有模型继续训练'])
        self.combo_train_mode.currentIndexChanged.connect(self.update_model_path_widgets)
        
        self.label_existing_model = QLabel("选择已有模型(.pth):")
        self.line_edit_existing_model = QLineEdit()
        self.button_existing_model = QPushButton("浏览")
        self.button_existing_model.clicked.connect(self.browse_existing_model)
        
        self.label_new_model = QLabel("新模型名称 (例如: my_model.pth):")
        self.line_edit_new_model = QLineEdit()
        
        self.button_process_batch = QPushButton("开始批处理和训练")
        self.button_process_batch.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        self.button_process_batch.clicked.connect(self.process_batch)

        # --- Layout ---
        main_layout = QVBoxLayout()
        
        # File selection layout
        file_selection_layout = QHBoxLayout()
        file_selection_layout.addWidget(self.line_edit_file)
        file_selection_layout.addWidget(self.button_file)

        # Group/Channel layout
        sampling_rate_layout = QHBoxLayout()
        sampling_rate_layout.addWidget(self.label_sampling_rate)
        sampling_rate_layout.addWidget(self.line_edit_sampling_rate)
        
        # Add to batch layout
        add_batch_layout = QHBoxLayout()
        add_batch_layout.addWidget(self.label_label)
        add_batch_layout.addWidget(self.combo_label)
        add_batch_layout.addWidget(self.button_add_to_batch, 1)

        main_layout.addWidget(self.label_file)
        main_layout.addLayout(file_selection_layout)
        main_layout.addLayout(sampling_rate_layout)
        main_layout.addLayout(add_batch_layout)
        
        main_layout.addSpacing(20)

        main_layout.addWidget(self.label_batch)
        main_layout.addWidget(self.table_batch)

        main_layout.addSpacing(20)

        main_layout.addWidget(self.label_train_setup)
        main_layout.addWidget(self.label_train_mode)
        main_layout.addWidget(self.combo_train_mode)
        main_layout.addWidget(self.label_existing_model)
        main_layout.addWidget(self.line_edit_existing_model)
        main_layout.addWidget(self.button_existing_model)
        main_layout.addWidget(self.label_new_model)
        main_layout.addWidget(self.line_edit_new_model)
        
        main_layout.addSpacing(10)
        main_layout.addWidget(self.button_process_batch)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.update_model_path_widgets() # Initial state

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择NPY文件", "", "NPY Files (*.npy)")
        if file_path:
            self.line_edit_file.setText(file_path)
    
    def browse_existing_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择已有模型", MODELS_DIR, "PyTorch Model Files (*.pth)")
        if file_path:
            self.line_edit_existing_model.setText(file_path)

    def update_model_path_widgets(self):
        is_continue_training = self.combo_train_mode.currentText() == '读取已有模型继续训练'
        self.label_existing_model.setVisible(is_continue_training)
        self.line_edit_existing_model.setVisible(is_continue_training)
        self.button_existing_model.setVisible(is_continue_training)
        
    def add_to_batch(self):
        file_path = self.line_edit_file.text()
        label = self.combo_label.currentText()
        
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", "请选择一个有效的NPY文件")
            return

        row_count = self.table_batch.rowCount()
        self.table_batch.insertRow(row_count)
        self.table_batch.setItem(row_count, 0, QTableWidgetItem(file_path))
        self.table_batch.setItem(row_count, 1, QTableWidgetItem(label))
        self.table_batch.setItem(row_count, 2, QTableWidgetItem("待处理"))
        
        print(f"Added to batch: {file_path} with label '{label}'")

    def process_batch(self):
        # --- 1. Validation ---
        if self.table_batch.rowCount() == 0:
            QMessageBox.warning(self, "警告", "批处理列表为空，请先添加文件。")
            return
            
        new_model_name = self.line_edit_new_model.text()
        if not new_model_name or not new_model_name.endswith('.pth'):
            QMessageBox.warning(self, "警告", "请输入有效的新模型名称，必须以 .pth 结尾。")
            return
        new_model_path = os.path.join(MODELS_DIR, new_model_name)

        # --- 2. Step 1: Process all NPY files to MAT files ---
        mat_file_list = []
        label_list_for_build = []
        
        try:
            sampling_rate_text = self.line_edit_sampling_rate.text()
            if not sampling_rate_text.isdigit() or int(sampling_rate_text) <= 0:
                QMessageBox.warning(self, "警告", "请输入一个有效的正整数作为采样率。")
                return
            sampling_rate = int(sampling_rate_text)

            QApplication.processEvents() # Update UI
            for row in range(self.table_batch.rowCount()):
                npy_path = self.table_batch.item(row, 0).text()
                label_str = self.table_batch.item(row, 1).text()

                self.table_batch.item(row, 2).setText("正在处理 (LCD-FastICA)...")
                QApplication.processEvents()

                # Create unique output path for the .mat file
                base_name = os.path.basename(npy_path)
                file_name_no_ext = os.path.splitext(base_name)[0]
                mat_file_name = f"{file_name_no_ext}_{int(time.time())}.mat"
                mat_output_path = os.path.join(ICA_RESULTS_DIR, mat_file_name)
                
                print(f"Processing {npy_path} -> {mat_output_path}")
                fast_ica_processing(npy_path, mat_output_path, sampling_rate)
                
                mat_file_list.append(mat_output_path)
                label_list_for_build.append(self.label_map[label_str])
                
                self.table_batch.item(row, 2).setText("ICA完成")
                QApplication.processEvents()
            
            print("All files processed to .mat format.")

            # --- 3. Step 2: Build Tensor Dataset from MAT files ---
            print("Building tensor dataset...")
            QMessageBox.information(self, "进度", "ICA处理完成，即将开始构建张量数据集。")
            build_tensor_data(mat_file_list, label_list_for_build, TENSOR_DATA_DIR)
            print("Tensor dataset built successfully.")

            # --- 4. Step 3: Train the model ---
            print("Starting model training...")
            QMessageBox.information(self, "进度", "数据集构建完成，即将开始训练模型。")
            
            train_mode = self.combo_train_mode.currentText()
            existing_model_path = self.line_edit_existing_model.text() if train_mode == '读取已有模型继续训练' else None
            
            # The train_model function needs to be adapted to this new flow
            # It no longer needs 'selected_label'
            train_model(
                folder_path=TENSOR_DATA_DIR, 
                train_mode=train_mode, 
                existing_model_path=existing_model_path, 
                new_model_path=new_model_path
            )
            
            QMessageBox.information(self, "成功", f"处理和训练全部完成！\n模型已保存至: {new_model_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
