echo "RGB ONLY"
echo python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-GRGB.py work_dirs/GRGBnet_Base/epoch_350.pth metrics=accuracy
echo "Greyscale Only"
echo python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-G.py work_dirs/GRGBnet_Base/epoch_350.pth metrics=accuracy
echo "Threshhold 50"
echo python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-50.py work_dirs/GRGBnet_Base/epoch_350.pth metrics=accuracy
echo "Threshhold 60"
echo python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-60.py work_dirs/GRGBnet_Base/epoch_350.pth metrics=accuracy
echo "Threshhold 70"
echo python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-70.py work_dirs/GRGBnet_Base/epoch_350.pth metrics=accuracy
echo "Threshhold 80"
echo python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-80.py work_dirs/GRGBnet_Base/epoch_350.pth metrics=accuracy
echo "Threshhold 90"
echo python3 tools/test.py configs/dynamic_networks/experimental-setups/GRGB-Net/GRGBnet-ImageNette-90.py work_dirs/GRGBnet_Base/epoch_350.pth metrics=accuracy