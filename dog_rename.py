import os

class ImageRename():
    def __init__(self):
        self.path = './data/train'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 12500

        for item in filelist:
            if 'dog' in item:
                src = os.path.join(self.path, item)
                dst = os.path.join(self.path, 'dog.' + str(i) + '.jpg')
                os.rename(src, dst)
                i = i + 1

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()