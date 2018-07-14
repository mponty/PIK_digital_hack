import pandas as pd

best_za = pd.read_csv('submission_final.csv')
best_c1 = pd.read_csv('c1imb3r_submit_max.csv')

mic_c = 0.55 
best = best_za.merge(best_c1, how = 'inner', on = 'id')

best['value'] = mic_c*best['value_x']+(1-mic_c)*best['value_y']
     
filename = f'blend_55_169_172.csv'
best[['id','value']].to_csv(filename, index = False)
