input_file = "/sdd/Dubaoset/src/Phong/Log/lat_lon_2021.csv"
output_file = "/sdd/Dubaoset/src/Phong/Log/lat_lon_2021_new.csv"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        # thay khoảng trắng bằng dấu phẩy
        new_line = line.replace("	", ",")
        f_out.write(new_line)

print("✅ Đã xử lý xong, file lưu tại:", output_file)
