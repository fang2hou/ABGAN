call activate fang
export PYTHONPATH=abganlibs:$PYTHONPATH
python experiments/sagan_21.py --cuda --ngpu=1
#python tools/original_level_generator/generator_competition.py