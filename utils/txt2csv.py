import csv
import os

if __name__ == '__main__':
    lst_path_lbl = []
    txt_file = "/media/dev01/MegaDataset/BeatNet/210907/256_60.0_480_0_0_0_1_0_0.8/output-4.010523/ds_train.txt"
    read_csv = open(txt_file, 'r')
    reader = csv.reader(read_csv)
    for rows in reader:
        lst_path_lbl.append(rows[0])

    read_csv.close()

    write_csv = open(txt_file, 'w')
    fields = ["id", "path", "check"]
    write = csv.writer(write_csv)
    write.writerow(fields)
    for pth in lst_path_lbl:
        write.writerow([os.path.basename(pth)[:-4], pth, False])

    write_csv.close()

    lst_path_lbl = []
    txt_file = "/media/dev01/MegaDataset/BeatNet/210907/256_60.0_480_0_0_0_1_0_0.8/output-4.010523/ds_eval.txt"
    read_csv = open(txt_file, 'r')
    reader = csv.reader(read_csv)
    for rows in reader:
        lst_path_lbl.append(rows[0])

    read_csv.close()

    write_csv = open(txt_file, 'w')
    fields = ["id", "path", "check"]
    write = csv.writer(write_csv)
    write.writerow(fields)
    for pth in lst_path_lbl:
        write.writerow([os.path.basename(pth)[:-4], pth, False])

    write_csv.close()
