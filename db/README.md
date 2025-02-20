# Database 

## Structure

This package contains all necessary files to run database queries.
All operators are located in the sub-package [operators](operators).
A criterion can be constructed from the [criteria.py](criteria.py) file.
Use [db.py](db.py) to establish a connection to a PostgreSQL Database. 

The file [structure.py](structure.py) contains the code to construct schemas (internal usage).  

```bash
📂
├── 🗎 README.md
├── 🗎 __init__.py
├── 🗎 criteria.py
├── 🗎 db.py
├── 🗎 structure.py
└── 📂 operators
    ├── 🗎 __init__.py
    ├── 🗎 Operator.py
    ├── 🗎 Dummy.py
    ├── 🗎 Aggregate.py
    ├── 🗎 Scan.py
    ├── 🗎 Select.py
    ├── 🗎 Project.py
    ├── 🗎 Join.py  
    ├── 🗎 Aggregate.py
    └── 🗎 Union.py
```

