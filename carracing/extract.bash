for i in `seq 1 64`;
do
  echo worker $i
  # on cloud:
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py &
  # on macbook for debugging:
  #python extract.py &
  sleep 1.0
done
