#python remote_fcgmm.py --config copyroom.yaml --path_results /media/fractal/T7/rss2023-resub/results --components 100 --inf_type cpu
#python remote_fcgmm.py --config copyroom.yaml --path_results /media/fractal/T7/rss2023-resub/results --components 200 --inf_type cpu
#python remote_fcgmm.py --config copyroom.yaml --path_results /media/fractal/T7/rss2023-resub/results --components 400 --inf_type cpu
#python remote_fcgmm.py --config copyroom.yaml --path_results /media/fractal/T7/rss2023-resub/results --components 800 --inf_type cpu
python remote_isogmm.py --config livingroom1.yaml --path_results /media/fractal/T7/rss2023-resub/results --bandwidth 0.05 --alpha 0.8
python remote_isogmm.py --config livingroom1.yaml --path_results /media/fractal/T7/rss2023-resub/results --bandwidth 0.05 --alpha 0.4
python remote_isogmm.py --config livingroom1.yaml --path_results /media/fractal/T7/rss2023-resub/results --bandwidth 0.05 --alpha 0.2
python remote_isogmm.py --config livingroom1.yaml --path_results /media/fractal/T7/rss2023-resub/results --bandwidth 0.05 --alpha 0.1
