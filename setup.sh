echo 'Making sure modular_rl is up to date'
cd /modular_rl && git pull
cd /ml_p5_capstone

echo 'Faking monitor'
xvfb-run -s "-screen 0 1400x900x24" bash
