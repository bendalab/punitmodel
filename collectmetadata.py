import numpy as np
from thunderlab.tabledata import TableData


def collect_metadata(models, data):
    # extract cell names from model table:
    model_data = TableData(models, sep=',')
    cells = model_data['cell']
    cells = np.unique(cells)
    # collect corresponding metadata:
    metadata = TableData()
    for cell in cells:
        md = data[data['cell'] == cell, :]
        if len(md) == 0:
            cell = '-'.join(cell.split('-')[:4])
            md = data[data['cell'] == cell, :]
        metadata.add(md[0, :], 0)
        metadata.fill_data()
    metadata.write(models.replace('.csv', '-metadata.csv'), delimiter=';')
   

if __name__ == '__main__':
    data = TableData('../punitdata/allcellsmetadata.csv', sep=';')
    for models in ['models_202106.csv', 'models.csv', 'models_old.csv']:
        collect_metadata(models, data)

