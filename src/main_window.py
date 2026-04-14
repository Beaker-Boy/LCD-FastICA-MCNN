import sys
import os
import time
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QLineEdit, QFileDialog, QVBoxLayout, QWidget, 
                             QMessageBox, QComboBox, QTableWidget, QTableWidgetItem,
                             QHBoxLayout, QHeaderView, QGroupBox, QProgressBar, QDialog, QCheckBox)
from PyQt5.QtCore import Qt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from lcd_fastica import process_signal_pipeline, fast_ica_processing
from build_tensor import build_tensor_data
from train_model import train_model
from evaluate import evaluate_single_model, compare_models

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
INTERMEDIATE_PLOTS_DIR = os.path.join(ICA_RESULTS_DIR, 'intermediate_plots')

# Ensure all directories exist
os.makedirs(ICA_RESULTS_DIR, exist_ok=True)
os.makedirs(TENSOR_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_PLOTS_DIR, exist_ok=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信号处理与模型训练")
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
        
        # Checkbox for plotting intermediate results
        self.checkbox_plot_intermediate = QCheckBox("绘制中间产物图线")
        self.checkbox_plot_intermediate.setToolTip("勾选后将保存分解/分离过程中的分量图线到 processed_data/ica_results/intermediate_plots 目录")
        self.checkbox_plot_intermediate.setChecked(False)
        processing_layout.addWidget(self.checkbox_plot_intermediate)
        
        processing_group.setLayout(processing_layout)
        
        self.button_add_to_batch = QPushButton("添加至批处理列表")
        self.button_add_to_batch.clicked.connect(self.add_to_batch)

        # --- Middle Level: Batch View ---
        self.label_batch = QLabel("2. 批处理列表:")
        self.table_batch = QTableWidget()
        self.table_batch.setColumnCount(3)
        self.table_batch.setHorizontalHeaderLabels(["NPY 文件路径", "标签", "状态"])
        # Fix: Add type assertion for horizontalHeader()
        header = self.table_batch.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_batch.setAlternatingRowColors(True)

        # --- Bottom Level: Training Setup and Execution ---
        self.label_train_setup = QLabel("3. 训练设置与执行:")
        
        # Model architecture selection
        self.label_model_arch = QLabel("选择模型架构:")
        self.combo_model_arch = QComboBox()
        from cnn_models import get_available_models
        available_models = ['MCNN'] + get_available_models()
        self.combo_model_arch.addItems(available_models)
        self.combo_model_arch.setToolTip("选择要训练的神经网络模型架构")
        
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
        main_layout.addWidget(self.label_model_arch)
        main_layout.addWidget(self.combo_model_arch)
        main_layout.addWidget(self.label_train_mode)
        main_layout.addWidget(self.combo_train_mode)
        main_layout.addWidget(self.label_existing_model)
        main_layout.addWidget(self.line_edit_existing_model)
        main_layout.addWidget(self.button_existing_model)
        main_layout.addWidget(self.label_new_model)
        main_layout.addWidget(self.line_edit_new_model)
        
        main_layout.addSpacing(10)
        main_layout.addWidget(self.button_process_batch)
        
        # Add evaluation section
        main_layout.addSpacing(20)
        eval_group = QGroupBox("模型评估")
        eval_layout = QHBoxLayout()
        
        self.button_evaluate = QPushButton("评估已训练模型")
        self.button_evaluate.setStyleSheet("background-color: lightblue; font-weight: bold;")
        self.button_evaluate.clicked.connect(self.open_evaluation_dialog)
        
        self.button_compare = QPushButton("对比多个模型")
        self.button_compare.setStyleSheet("background-color: lightcoral; font-weight: bold;")
        self.button_compare.clicked.connect(self.open_comparison_dialog)
        
        eval_layout.addWidget(self.button_evaluate)
        eval_layout.addWidget(self.button_compare)
        
        eval_group.setLayout(eval_layout)
        main_layout.addWidget(eval_group)
        
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
        # Store the processing methods list in userData of the status item (column 2)
        status_item = QTableWidgetItem("待处理")
        # Fix: Add type assertion for Qt.UserRole
        status_item.setData(Qt.UserRole, selected_methods)  # type: ignore[arg-type]
        self.table_batch.setItem(row_count, 2, status_item)
        
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
                # Fix: Add None checks for table items
                file_item = self.table_batch.item(row, 0)
                label_item = self.table_batch.item(row, 1)
                status_item = self.table_batch.item(row, 2)
                
                if file_item is None or label_item is None or status_item is None:
                    logger.error(f"Invalid table row {row}, skipping...")
                    continue
                
                npy_path = file_item.text()
                label_str = label_item.text()
                
                # Retrieve the processing methods for this file
                # Fix: Add type assertion for Qt.UserRole and data retrieval
                selected_methods = status_item.data(Qt.UserRole) if status_item else []  # type: ignore[arg-type]
                
                # Ensure selected_methods is a valid list
                if not selected_methods or not isinstance(selected_methods, list):
                    selected_methods = []
                
                pipeline_text = ' + '.join(selected_methods) if selected_methods else "未指定方法"
                # Fix: Add None check before calling setText
                if status_item is not None:
                    status_item.setText(f"正在处理 ({pipeline_text})...")
                
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
                logger.info(f"Processing pipeline: {' + '.join(selected_methods) if selected_methods else 'None'}")
                
                try:
                    # Determine plot settings
                    plot_intermediate = self.checkbox_plot_intermediate.isChecked()
                    plot_save_dir = INTERMEDIATE_PLOTS_DIR if plot_intermediate else None
                    
                    # Use the new flexible pipeline with progress callback
                    process_signal_pipeline(
                        file_path=npy_path,
                        output_path=mat_output_path,
                        sampling_rate=sampling_rate,
                        processing_methods=selected_methods,
                        max_samples=max_samples,
                        progress_callback=self.progress_callback,
                        plot_intermediate=plot_intermediate,
                        plot_save_dir=plot_save_dir
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
                
                # Fix: Add None check before calling setText
                status_item_updated = self.table_batch.item(row, 2)
                if status_item_updated is not None:
                    status_item_updated.setText("ICA 完成")
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
            model_arch = self.combo_model_arch.currentText()  # Get selected model architecture
            
            train_model(
                folder_path=TENSOR_DATA_DIR, 
                train_mode=train_mode, 
                existing_model_path=existing_model_path, 
                new_model_path=new_model_path,
                model_arch=model_arch  # Pass model architecture
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
    
    def open_evaluation_dialog(self):
        """打开单模型评估对话框"""
        dialog = EvaluationDialog(self, mode='single')
        if dialog.exec_():
            logger.info("模型评估完成")
    
    def open_comparison_dialog(self):
        """打开多模型对比对话框"""
        dialog = EvaluationDialog(self, mode='comparison')
        if dialog.exec_():
            logger.info("模型对比完成")


class EvaluationDialog(QDialog):
    """模型评估对话框"""
    
    def __init__(self, parent=None, mode='single'):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle("模型评估" if mode == 'single' else "模型对比")
        self.setMinimumSize(600, 400)
        
        # Set up directory for results
        self.results_dir = os.path.join(PROJECT_ROOT_DIR, '..', 'results', 'evaluation')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.selected_files = []
        self.data_dir = TENSOR_DATA_DIR
        
        self.initUI()
    
    def initUI(self):
        """初始化 UI"""
        layout = QVBoxLayout()
        
        # 说明标签
        if self.mode == 'single':
            info_text = "请选择要评估的模型文件 (.pth)\n\n评估完成后，结果将保存在:\n" + self.results_dir
        else:
            info_text = "请选择要对比的多个模型文件 (.pth)\n\n可以使用 Ctrl/Cmd 键选择多个文件\n\n对比结果将保存在:\n" + self.results_dir
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info_label)
        
        layout.addSpacing(20)
        
        # 模型文件选择
        model_select_group = QGroupBox("模型文件选择")
        model_select_layout = QVBoxLayout()
        
        self.model_list_widget = QTableWidget()
        self.model_list_widget.setColumnCount(2)
        self.model_list_widget.setHorizontalHeaderLabels(["模型文件", "大小"])
        header = self.model_list_widget.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.model_list_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.model_list_widget.setAlternatingRowColors(True)
        model_select_layout.addWidget(self.model_list_widget)
        
        browse_model_btn = QPushButton("浏览模型文件 (.pth)")
        browse_model_btn.clicked.connect(self.browse_models)
        model_select_layout.addWidget(browse_model_btn)
        
        model_select_group.setLayout(model_select_layout)
        layout.addWidget(model_select_group)
        
        layout.addSpacing(10)
        
        # 数据目录选择
        data_dir_group = QGroupBox("数据集目录")
        data_dir_layout = QHBoxLayout()
        
        self.data_dir_edit = QLineEdit(self.data_dir)
        data_dir_layout.addWidget(self.data_dir_edit)
        
        browse_data_btn = QPushButton("浏览")
        browse_data_btn.clicked.connect(self.browse_data_dir)
        data_dir_layout.addWidget(browse_data_btn)
        
        data_dir_group.setLayout(data_dir_layout)
        layout.addWidget(data_dir_group)
        
        layout.addSpacing(20)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.btn_ok = QPushButton("确定")
        self.btn_ok.setStyleSheet("background-color: lightgreen; font-weight: bold; padding: 8px;")
        self.btn_ok.clicked.connect(self.accept)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setStyleSheet("padding: 8px;")
        self.btn_cancel.clicked.connect(self.reject)
        
        button_layout.addWidget(self.btn_ok)
        button_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.update_button_state()
    
    def browse_models(self):
        """浏览模型文件"""
        if self.mode == 'single':
            # 单文件选择
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "选择模型文件",
                MODELS_DIR,
                "PyTorch Model Files (*.pth)"
            )
            if file_path:
                self._add_model_to_list(file_path)
        else:
            # 多文件选择
            files, _ = QFileDialog.getOpenFileNames(
                self, 
                "选择多个模型文件",
                MODELS_DIR,
                "PyTorch Model Files (*.pth)"
            )
            if files:
                for file_path in files:
                    self._add_model_to_list(file_path)
        
        self.update_button_state()
    
    def _add_model_to_list(self, file_path):
        """添加模型到列表"""
        # 检查是否已存在
        for row in range(self.model_list_widget.rowCount()):
            item = self.model_list_widget.item(row, 0)
            if item is not None and item.text() == file_path:
                return  # 已存在，不重复添加
        
        row_count = self.model_list_widget.rowCount()
        self.model_list_widget.insertRow(row_count)
        
        # 文件名
        file_name = os.path.basename(file_path)
        self.model_list_widget.setItem(row_count, 0, QTableWidgetItem(file_path))
        
        # 文件大小
        try:
            size_bytes = os.path.getsize(file_path)
            size_str = f"{size_bytes / (1024*1024):.2f} MB"
        except:
            size_str = "未知"
        
        self.model_list_widget.setItem(row_count, 1, QTableWidgetItem(size_str))
    
    def browse_data_dir(self):
        """浏览数据目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择数据集目录",
            self.data_dir
        )
        if dir_path:
            self.data_dir_edit.setText(dir_path)
    
    def update_button_state(self):
        """更新按钮状态"""
        has_files = self.model_list_widget.rowCount() > 0
        self.btn_ok.setEnabled(has_files)
        
        if has_files:
            self.btn_ok.setStyleSheet("background-color: lightgreen; font-weight: bold; padding: 8px;")
        else:
            self.btn_ok.setStyleSheet("background-color: lightgray; padding: 8px;")
    
    def get_selected_models(self):
        """获取选中的模型文件列表"""
        models = []
        for row in range(self.model_list_widget.rowCount()):
            item = self.model_list_widget.item(row, 0)
            if item is not None:
                file_path = item.text()
                models.append(file_path)
        return models
    
    def accept(self):
        """处理确认按钮"""
        selected_models = self.get_selected_models()
        data_dir = self.data_dir_edit.text()
        
        if not selected_models:
            QMessageBox.warning(self, "警告", "请至少选择一个模型文件")
            return
        
        if not os.path.exists(data_dir):
            QMessageBox.warning(self, "警告", f"数据集目录不存在：{data_dir}")
            return
        
        try:
            if self.mode == 'single':
                # 单模型评估
                model_path = selected_models[0]
                save_dir = os.path.join(self.results_dir, os.path.basename(model_path).replace('.pth', ''))
                
                QMessageBox.information(
                    self, 
                    "开始评估", 
                    f"正在评估模型：{os.path.basename(model_path)}\n\n结果将保存至：{save_dir}"
                )
                
                # 隐藏对话框并执行评估
                self.hide()
                evaluate_single_model(model_path, data_dir, save_dir)
                
                QMessageBox.information(
                    self, 
                    "评估完成", 
                    f"评估已完成！\n\n结果保存位置:\n{save_dir}\n\n查看以下文件:\n- evaluation_results.json (详细指标)\n- confusion_matrix.png (混淆矩阵)\n- tsne_visualization.png (t-SNE 图)"
                )
            
            else:
                # 多模型对比
                save_dir = os.path.join(self.results_dir, f"comparison_{len(selected_models)}_models")
                
                QMessageBox.information(
                    self, 
                    "开始对比", 
                    f"正在对比 {len(selected_models)} 个模型\n\n结果将保存至：{save_dir}"
                )
                
                # 隐藏对话框并执行对比
                self.hide()
                compare_models(selected_models, data_dir, save_dir)
                
                QMessageBox.information(
                    self, 
                    "对比完成", 
                    f"模型对比已完成！\n\n结果保存位置:\n{save_dir}\n\n查看以下文件:\n- model_comparison.json (对比结果)\n- 各模型的详细评估报告"
                )
            
            super().accept()
            
        except Exception as e:
            logger.error(f"评估失败：{str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"评估过程中发生错误:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
