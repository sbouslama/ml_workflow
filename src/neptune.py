# XGBoost + Neptune integration

# Before you start

## Install dependencies

import neptune
import os

# Set project
neptune.init('',
             api_token=os.getenv("NEPTUNE_API_TOKEN")
)

