import importlib
import inspect
import sys
from pathlib import Path

filename = 'ModelData/d4a81140-6204-4aeb-be68-0208470edeff/Model.py'
path_pyfile = Path(filename)
sys.path.append(str(path_pyfile.parent))
mod_path = str(path_pyfile).replace('\\', '.').strip('.py')
mysterious = importlib.import_module(mod_path)

for name_local in dir(mysterious):
    print('name local ' + str(name_local))
    if inspect.isclass(getattr(mysterious, name_local)):
        print(f'{name_local} is a class')
        MysteriousClass = getattr(mysterious, name_local)
        mysterious_object = MysteriousClass()
        print(mysterious_object)