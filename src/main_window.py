import sys
import os
import time
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QLineEdit, QFileDialog, QVBoxLayout, QWidget, 
                             QMessageBox, QComboBox, QTableWidget, QTableWidgetItem,
                             QHBoxLayout, QHeaderView, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from lcd_fastica import process_signal_pipeline, fast_ica_processing
from build_tensor import build_tensor_data
from train_model import train_model

# Define available processing methods
DECOMPOSITION_METHODS = ['LCD', 'VMD', 'EEMD', 'LMD']
POST_PROCESSING_METHODS = ['None', 'FastICA']

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
        self.label_file = QLabel("1. 选择 NPY 文件:")
        self.line_edit_file = QLineEdit()
        self.button_file = QPushButton("浏览")
        self.button_file.clicked.connect(self.browse_file)
        
        self.label_sampling_rate = QLabel("采样率 (Hz):")
        self.line_edit_sampling_rate = QLineEdit("20000") # Default value

        self.label_max_samples = QLabel("最大采样点数:")
        self.line_edit_max_samples = QLineEdit("9142857") # Default value
        
        self.label_label = QLabel("选择标签:")
        self.combo_label = QComboBox()
        self.combo_label.addItems(self.label_map.keys())
        
        # --- Processing Method Selection (Redesigned for single decomposition + optional FastICA) ---
        processing_group = QGroupBox("信号处理方法配置（一种分解方法 + 可选 FastICA）")
        processing_layout = QVBoxLayout()
        
        # First combo: Select ONE decomposition method
        decomp_layout = QHBoxLayout()
        decomp_label = QLabel("分解方法:")
        self.combo_decomposition = QComboBox()
        self.combo_decomposition.addItems(DECOMPOSITION_METHODS)
        self.combo_decomposition.setToolTip("选择一种信号分解方法")
        decomp_layout.addWidget(decomp_label)
        decomp_layout.addWidget(self.combo_decomposition, 1)
        processing_layout.addLayout(decomp_layout)
        
        # Second combo: Optional FastICA
        ica_layout = QHBoxLayout()
        ica_label = QLabel("后处理:")
        self.combo_ica = QComboBox()
        self.combo_ica.addItems(['无', 'FastICA'])
        self.combo_ica.setToolTip("可选择是否使用 FastICA 进行独立成分分析")
        ica_layout.addWidget(ica_label)
        ica_layout.addWidget(self.combo_ica, 1)
        processing_layout.addLayout(ica_layout)
        
        processing_group.setLayout(processing_layout)
        
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
        
        # Max samples layout
        max_samples_layout = QHBoxLayout()
        max_samples_layout.addWidget(self.label_max_samples)
        max_samples_layout.addWidget(self.line_edit_max_samples)

        # Add to batch layout
        add_batch_layout = QHBoxLayout()
        add_batch_layout.addWidget(self.label_label)
        add_batch_layout.addWidget(self.combo_label)
        add_batch_layout.addWidget(self.button_add_to_batch, 1)

        main_layout.addWidget(self.label_file)
        main_layout.addLayout(file_selection_layout)
        main_layout.addLayout(sampling_rate_layout)
        main_layout.addLayout(max_samples_layout)
        main_layout.addLayout(add_batch_layout)
        
        # Add processing group
        main_layout.addWidget(processing_group)
        
        main_layout.addSpacing(20)

        main_layout.addWidget(self.label_batch)
        main_layout.addWidget(self.table_batch)
        
        # Progress bar for batch processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("处理进度：")
        self.progress_label.setVisible(False)
        main_layout.addWidget(self.progress_label)
        main_layout.addWidget(self.progress_bar)

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
            QMessageBox.warning(self, "警告", "请选择一个有效的 NPY 文件")
            return
        
        # Collect selected processing methods from new UI
        decomp_method = self.combo_decomposition.currentText()
        ica_method = self.combo_ica.currentText()
        
        # Build processing methods list
        selected_methods = [decomp_method]
        if ica_method == 'FastICA':
            selected_methods.append('FastICA')
        
        pipeline_desc = ' + '.join(selected_methods)

        row_count = self.table_batch.rowCount()
        self.table_batch.insertRow(row_count)
        self.table_batch.setItem(row_count, 0, QTableWidgetItem(file_path))
        self.table_batch.setItem(row_count, 1, QTableWidgetItem(label))
        # Store the processing methods list in userData
        item = QTableWidgetItem(pipeline_desc)
        item.setData(Qt.UserRole, selected_methods)
        self.table_batch.setItem(row_count, 2, QTableWidgetItem("待处理"))
        
        logger.info(f"Added to batch: {file_path} with label '{label}' and processing pipeline: {pipeline_desc}")

    def progress_callback(self, current_step, total_steps, message):
        """Progress callback function for signal processing"""
        # Update progress bar if available
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(int((current_step / max(total_steps, 1)) * 100))
            self.progress_label.setText(f"处理进度：{message}")
            QApplication.processEvents()
    
    def process_batch(self):
        # Show progress bar
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_bar.setValue(0)
        
        try:
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
            
            sampling_rate_text = self.line_edit_sampling_rate.text()
            if not sampling_rate_text.isdigit() or int(sampling_rate_text) <= 0:
                QMessageBox.warning(self, "警告", "请输入一个有效的正整数作为采样率。")
                return
            sampling_rate = int(sampling_rate_text)

            max_samples_text = self.line_edit_max_samples.text()
            if not max_samples_text.isdigit() or int(max_samples_text) <= 0:
                QMessageBox.warning(self, "警告", "请输入一个有效的正整数作为最大采样点数。")
                return
            max_samples = int(max_samples_text)

            QApplication.processEvents()
            total_files = self.table_batch.rowCount()
            
            for row in range(total_files):
                npy_path = self.table_batch.item(row, 0).text()
                label_str = self.table_batch.item(row, 1).text()
                
                # Retrieve the processing methods for this file
                status_item = self.table_batch.item(row, 2)
                selected_methods = status_item.data(Qt.UserRole) if status_item else []

                pipeline_text = ' + '.join(selected_methods)
                self.table_batch.item(row, 2).setText(f"正在处理 ({pipeline_text})...")
                
                # Update global progress
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.setValue(int((row / total_files) * 100))
                    self.progress_label.setText(f"文件 {row+1}/{total_files}: {os.path.basename(npy_path)}")
                QApplication.processEvents()

                # Create unique output path for the .mat file
                base_name = os.path.basename(npy_path)
                file_name_no_ext = os.path.splitext(base_name)[0]
                mat_file_name = f"{file_name_no_ext}_{int(time.time())}.mat"
                mat_output_path = os.path.join(ICA_RESULTS_DIR, mat_file_name)
                
                logger.info(f"\nProcessing {npy_path} -> {mat_output_path}")
                logger.info(f"Processing pipeline: {' + '.join(selected_methods)}")
                
                try:
                    # Use the new flexible pipeline with progress callback
                    process_signal_pipeline(
                        file_path=npy_path,
                        output_path=mat_output_path,
                        sampling_rate=sampling_rate,
                        processing_methods=selected_methods,
                        max_samples=max_samples,
                        progress_callback=self.progress_callback
                    )
                except ImportError as e:
                    logger.error(f"依赖库缺失：{str(e)}")
                    QMessageBox.critical(self, "错误", 
                        f"处理文件 {npy_path} 时依赖库缺失:\n{str(e)}\n\n请安装所需依赖后重试。")
                    return
                except ValueError as e:
                    logger.error(f"参数验证失败：{str(e)}")
                    QMessageBox.critical(self, "错误", f"处理方法配置错误:\n{str(e)}")
                    return
                except RuntimeError as e:
                    logger.error(f"处理失败：{str(e)}")
                    QMessageBox.critical(self, "错误", f"处理文件 {npy_path} 时出错:\n{str(e)}")
                    return
                
                mat_file_list.append(mat_output_path)
                label_list_for_build.append(self.label_map[label_str])
                
                self.table_batch.item(row, 2).setText("ICA 完成")
                QApplication.processEvents()
            
            logger.info("All files processed to .mat format.")

            # --- 3. Step 2: Build Tensor Dataset from MAT files ---
            logger.info("Building tensor dataset...")
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(75)
                self.progress_label.setText("构建张量数据集...")
            QApplication.processEvents()
            
            build_tensor_data(mat_file_list, label_list_for_build, TENSOR_DATA_DIR)
            logger.info("Tensor dataset built successfully.")

            # --- 4. Step 3: Train the model ---
            logger.info("Starting model training...")
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(90)
                self.progress_label.setText("训练模型中...")
            QApplication.processEvents()
            
            train_mode = self.combo_train_mode.currentText()
            existing_model_path = self.line_edit_existing_model.text() if train_mode == '读取已有模型继续训练' else None
            
            train_model(
                folder_path=TENSOR_DATA_DIR, 
                train_mode=train_mode, 
                existing_model_path=existing_model_path, 
                new_model_path=new_model_path
            )
            
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(100)
                self.progress_label.setText("全部完成!")
            
            QMessageBox.information(self, "成功", f"处理和训练全部完成！\n模型已保存至：{new_model_path}")

        except Exception as e:
            logger.error(f"处理过程中发生错误：{str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"处理过程中发生错误：{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Hide progress bar after completion or error
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)
                self.progress_label.setVisible(False)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
