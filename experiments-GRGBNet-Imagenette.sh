set -x
echo "RGB ONLY"
python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-RGB.py work_dirs/GRGBnet_Base/epoch_350.pth --metrics=accuracy
echo "Greyscale Only"
python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-G.py work_dirs/GRGBnet_Base/epoch_350.pth --metrics=accuracy
echo "Threshhold 50"
python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-50.py work_dirs/GRGBnet_Base/epoch_350.pth --metrics=accuracy
echo "Threshhold 60"
python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-60.py work_dirs/GRGBnet_Base/epoch_350.pth --metrics=accuracy
echo "Threshhold 70"
python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-70.py work_dirs/GRGBnet_Base/epoch_350.pth --metrics=accuracy
echo "Threshhold 80"
python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-80.py work_dirs/GRGBnet_Base/epoch_350.pth --metrics=accuracy
echo "Threshhold 90"
python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-90.py work_dirs/GRGBnet_Base/epoch_350.pth --metrics=accuracy
set +x
