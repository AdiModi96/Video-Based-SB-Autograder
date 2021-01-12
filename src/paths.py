import os
import inspect

current_file_path = inspect.getfile(inspect.currentframe())
project_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(current_file_path)))
src_folder_path = os.path.join(project_folder_path, 'src')
resrc_folder_path = os.path.join(project_folder_path, 'resrc')
data_folder_path = os.path.join(project_folder_path, 'data')
