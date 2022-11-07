# local Packages
from cnn_process.TestModel import performance_metrics, append_matrix
from cnn_process.load.load_main import PhysNet

name_no_face = "noFaceListAllVideos_" + data_split + ".pkl"
name_deleted_videos = "delete_videos_" + data_split + ".pkl"
list_path = config['variblesPath']
path_deleted_videos = os.path.join(list_path, name_deleted_videos)
path_no_face = os.path.join(list_path, name_no_face)

open_file = open(path_no_face, "rb")
noFaceListAllVideos = pickle.load(open_file)
open_file.close()

open_file = open(path_deleted_videos, "rb")
delete_videos = pickle.load(open_file)
open_file.close()



try:
    print("Resluts with complete test dataset")
    MAE, RMSE, STD = performance_metrics.eval_model_fft(BVP_label_all, rPPG_all, config)
    print(f"Test fft MAE: {MAE:.3f}" + f" Test RMSE: {RMSE:.3f}")
    MAE, RMSE, STD = performance_metrics.eval_model(BVP_label_all, rPPG_all, config)
    print(f"Test MAE: {MAE:.3f}" + f" Test RMSE: {RMSE:.3f}")
except Exception:
    print("Could not determine pulse for given signal.")