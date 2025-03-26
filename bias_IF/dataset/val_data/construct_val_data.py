import pandas as pd

crows_path = '/data4/whr/dxl/bias_IF/data/val_data/crows_data.csv'
seat_path = '/data4/whr/dxl/bias_IF/data/val_data/seat_data.csv'
stereo_path = '/data4/whr/dxl/bias_IF/data/val_data/stereoset_data.csv'

crows_col = pd.read_csv(crows_path).iloc[:, 0]
seat_col = pd.read_csv(seat_path).iloc[:, 0]
stereo_col = pd.read_csv(stereo_path).iloc[:, 0]

pd.concat([crows_col, seat_col]).to_csv('/data4/whr/dxl/bias_IF/data/val_data/val_stereoset.csv', index=False)

pd.concat([crows_col, stereo_col]).to_csv('/data4/whr/dxl/bias_IF/data/val_data/val_seat.csv', index=False)

pd.concat([seat_col, stereo_col]).to_csv('/data4/whr/dxl/bias_IF/data/val_data/val_crows.csv', index=False)

print("合并完成！生成了三个新文件：")
print("1. val_stereoset.csv (crows + seat)")
print("2. val_seat.csv (crows + stereoset)")
print("3. val_crows.csv (seat + stereoset)")