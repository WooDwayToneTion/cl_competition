import sqlite3
import os
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO


class SQLDataset(Dataset):
    def __init__(self, db_path, table_name, task_name, train_val=True, transform=None):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.task_name = task_name
        self.train_val = train_val
        self.transform = transform

        self.conn = None
        self.establish_conn()
        self.cursor.execute(f"select max(rowid) from {self.table_name}")
        self.nums = self.cursor.fetchall()[0][0]

    def __getitem__(self, index):
        self.establish_conn()
        self.cursor.execute(f"select * from {self.table_name} where rowid=?", (index + 1,))

        if self.train_val:
            img_bytes, label, task_name = self.cursor.fetchone()
        else:
            img_bytes, task_name = self.cursor.fetchone()

        assert str(task_name) == self.task_name
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        task_name = str(task_name)
        if self.transform:
            img = self.transform(img)
        if self.train_val:
            return img, label, task_name
        else:
            return img, task_name

    def __len__(self):
        return self.nums

    def establish_conn(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, cached_statements=1024)
            self.cursor = self.conn.cursor()
        return self

    def close_conn(self):
        if self.conn is not None:
            self.cursor.close()
            self.conn.close()
            del self.conn
            self.conn = None
        return self


def get_this_dataset(root, task_name, train_trans=None, val_trans=None, test_trans=None, training=True):
    dataset_name = os.path.split(root)[1]
    assert dataset_name in ['4splitDomains', '10splitTasks'], 'Error: invalid dataset name!'
    table_name_ = 'CL4SplitDomains' if dataset_name == '4splitDomains' else 'CL10SplitTasks'

    if training:
        task_train_path = os.path.join(root, 'train_{}.db'.format(task_name))
        table_name = table_name_ + f'train_{task_name}'
        print(table_name)
        train_dataset = SQLDataset(db_path=task_train_path, table_name=table_name, task_name=task_name, transform=train_trans)
        train_dataset.close_conn()

        task_val_path = os.path.join(root, 'val_{}.db'.format(task_name))
        table_name = table_name_ + f'val_{task_name}'
        val_dataset = SQLDataset(db_path=task_val_path, table_name=table_name, task_name=task_name, transform=val_trans)
        val_dataset.close_conn()

        return train_dataset, val_dataset
    else:
        task_test_path = os.path.join(root, 'test_{}.db'.format(task_name))
        table_name = table_name_ + f'test_{task_name}'
        test_dataset = SQLDataset(db_path=task_test_path, table_name=table_name, train_val=False, task_name=task_name, transform=test_trans)
        test_dataset.close_conn()
        return test_dataset


if __name__ == "__main__":
    # 读取 4splitDomains 样例
    # root 建议填写绝对路径
    HOME = "/kaggle/input/continual-learning-data/continual_learning_data/"
    train_dataset, val_dataset = get_this_dataset(root=HOME + '4splitDomains', task_name='3')
    for record in train_dataset:
        print(f"本训练任务包含{train_dataset.__len__()}条记录")
        print(record)
        break

    # 读取 10splitTasks 样例
    train_dataset, val_dataset = get_this_dataset(root=HOME + '10splitTasks', task_name='6')
    for record in val_dataset:
        print(f"本验证任务包含{val_dataset.__len__()}条记录")
        print(record)
        break