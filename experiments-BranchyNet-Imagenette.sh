CPT_FILE=work_dirs/BranchyNet-ImageNette2/epoch_65.pth
LOG_FILE=work_dirs/BranchyNet-ImageNette2/Experiments/experiment_log.txt

echo "Experiments started for BranchyNetImagenette2" 

echo "Treshhold Exit1: 0.50"
echo "Treshhold Exit1: 0.50" >> $LOG_FILE
echo "Treshhold Exit2: 0.50"
echo "Treshhold Exit2: 0.50" >> $LOG_FILE

CONFIG_FOLDER=configs/dynamic_networks/experimental-setups/BranchyNet-Imagenette2/th1-50-th2-50
OUT_DIR=work_dirs/BranchyNet-ImageNette2/Experiments/th1-50-th2-50

echo "Configuration: 001" 
echo "Configuration: 001" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/001_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet001.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 010" 
echo "Configuration: 010" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/010_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet010.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 011"
echo "Configuration: 011" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/011_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet011.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 100" 
echo "Configuration: 100" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/100_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet100.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 101" 
echo "Configuration: 101" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/101_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet101.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 110" 
echo "Configuration: 110" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/110_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 111"
echo "Configuration: 111" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/111_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE

echo "--------------------------------------------- /n"

echo "Treshhold Exit1: 0.80"
echo "Treshhold Exit1: 0.80" >> $LOG_FILE
echo "Treshhold Exit2: 0.60"
echo "Treshhold Exit2: 0.60" >> $LOG_FILE

CONFIG_FOLDER=configs/dynamic_networks/experimental-setups/BranchyNet-Imagenette2/th1-80-th2-60
OUT_DIR=work_dirs/BranchyNet-ImageNette2/Experiments/th1-80-th2-60

echo "Configuration: 001" 
echo "Configuration: 001" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/001_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet001.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 010" 
echo "Configuration: 010" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/010_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet010.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 011"
echo "Configuration: 011" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/011_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet011.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 100" 
echo "Configuration: 100" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/100_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet100.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 101" 
echo "Configuration: 101" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/101_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet101.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 110" 
echo "Configuration: 110" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/110_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 111"
echo "Configuration: 111" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/111_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE

echo "--------------------------------------------- /n"

echo "Treshhold Exit1: 0.80"
echo "Treshhold Exit1: 0.80" >> $LOG_FILE
echo "Treshhold Exit2: 0.70"
echo "Treshhold Exit2: 0.70" >> $LOG_FILE

CONFIG_FOLDER=configs/dynamic_networks/experimental-setups/BranchyNet-Imagenette2/th1-80-th2-70
OUT_DIR=work_dirs/BranchyNet-ImageNette2/Experiments/th1-80-th2-70

echo "Configuration: 001" 
echo "Configuration: 001" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/001_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet001.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 010" 
echo "Configuration: 010" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/010_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet010.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 011"
echo "Configuration: 011" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/011_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet011.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 100" 
echo "Configuration: 100" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/100_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet100.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 101" 
echo "Configuration: 101" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/101_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet101.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 110" 
echo "Configuration: 110" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/110_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 111"
echo "Configuration: 111" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/111_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE

echo "--------------------------------------------- /n"

echo "Treshhold Exit1: 0.90"
echo "Treshhold Exit1: 0.90" >> $LOG_FILE
echo "Treshhold Exit2: 0.60"
echo "Treshhold Exit2: 0.60" >> $LOG_FILE

CONFIG_FOLDER=configs/dynamic_networks/experimental-setups/BranchyNet-Imagenette2/th1-90-th2-60
OUT_DIR=work_dirs/BranchyNet-ImageNette2/Experiments/th1-90-th2-60

echo "Configuration: 001" 
echo "Configuration: 001" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/001_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet001.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 010" 
echo "Configuration: 010" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/010_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet010.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 011"
echo "Configuration: 011" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/011_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet011.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 100" 
echo "Configuration: 100" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/100_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet100.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 101" 
echo "Configuration: 101" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/101_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet101.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 110" 
echo "Configuration: 110" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/110_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 111"
echo "Configuration: 111" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/111_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE

echo "--------------------------------------------- /n"

echo "Treshhold Exit1: 0.90"
echo "Treshhold Exit1: 0.90" >> $LOG_FILE
echo "Treshhold Exit2: 0.70"
echo "Treshhold Exit2: 0.70" >> $LOG_FILE

CONFIG_FOLDER=configs/dynamic_networks/experimental-setups/BranchyNet-Imagenette2/th1-50-th2-50
OUT_DIR=work_dirs/BranchyNet-ImageNette2/Experiments/th1-50-th2-50

echo "Configuration: 001" 
echo "Configuration: 001" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/001_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet001.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 010" 
echo "Configuration: 010" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/010_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet010.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 011"
echo "Configuration: 011" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/011_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet011.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 100" 
echo "Configuration: 100" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/100_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet100.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 101" 
echo "Configuration: 101" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/101_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet101.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 110" 
echo "Configuration: 110" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/110_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE
echo "Configuration: 111"
echo "Configuration: 111" >> $LOG_FILE
START=date+%s
python3 tools/test.py $CONFIG_FOLDER/111_BranchyNet-Imagenette2.py \
        $CPT_FILE --metrics=accuracy \
        --out=$OUT_DIR/val_branchynet110.json >> $LOG_FILE
echo "Wallclock Time elapsed:" | date +%s - $START
echo "Wallclock Time elapsed:" | date +%s - $START >> $LOG_FILE

