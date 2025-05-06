from utils.ec57_test import ec57_eval, del_result, bxb_eval, del_result2, ec57_eval_event

# DB = [
#     ['european-st-t-database-1.0.0', 'atra', 'stca'],
#     ['european-st-t-database-1.0.0', 'atrb', 'stcb'],
# ]
#
# physionet_directory = '/home/dongbui-pc/Downloads/'
# output_ec57_directory = '/home/dongbui-pc/Desktop/escdb_log/'

DB = [
    ['european-st-t-database-1.0.0', 'atra', 'stca'],
    # ['european-st-t-database-1.0.0', 'atrb', 'stcb'],
]

physionet_directory = '/home/dongbui-pc/Desktop/escdb_log/'
output_ec57_directory = '/home/dongbui-pc/Desktop/escdb_log/'

for db in DB:
    # EC57 Eval-Full db
    ec57_eval(db[0],
              output_ec57_directory,
              physionet_directory,
              db[1],
              db[1],
              None,
              db[2],
              )