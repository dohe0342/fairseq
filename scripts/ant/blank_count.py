filename_list = ["d2v_vanilla_blank.txt", "d2v_mba_blank.txt"]

for filename in filename_list:
    file = open(filename, 'r').readlines()
    all_blank = 0
    all_length = 0
    for line in file[4:-4]:
        blank, length = line.split(' ')
        blank = int(blank)
        length = int(length.replace('\n', ''))
        all_blank += blank
        all_length += length

    print(filename, all_blank, all_length)
