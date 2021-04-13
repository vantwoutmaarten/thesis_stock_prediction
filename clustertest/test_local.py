import neptune

# this is a windows api token.

neptune.init(project_qualified_name='mavantwout/Cluster', # change this to your `workspace_name/project_name`
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0='
            )

neptune.create_experiment()

import numpy as np
from time import sleep

neptune.log_metric('single_metric', 0.62)

for i in range(100):
    sleep(0.2) # to see logging live
    neptune.log_metric('random_training_metric', i * np.random.random())
    neptune.log_metric('other_random_training_metric', 0.5 * i * np.random.random())

neptune.stop()