import xgboost as xgb
import numpy as np

def load_xgboost_model(filename):
    # Load model từ file
    model = xgb.Booster()
    model.load_model('XGBoost.json')
    return model

def make_prediction(model, data):
    # Chuyển đổi dữ liệu đầu vào thành DMatrix, format dữ liệu yêu cầu của XGBoost
    dmatrix_data = xgb.DMatrix(data)
    # Thực hiện dự đoán
    predictions = model.predict(dmatrix_data)
    return predictions

if __name__ == "__main__":
    # Đường dẫn tới file model
    model_filename = 'XGBoost.json'
    # Load model
    model = load_xgboost_model(model_filename)
    
    # Tạo dữ liệu đầu vào mẫu (giả sử model yêu cầu 3 features)
    sample_input = np.array([[1.5, 2.5, 3.5]])
    
    # Thực hiện dự đoán
    predictions = make_prediction(model, sample_input)
    
    # In kết quả dự đoán
    print("Predictions:", predictions)