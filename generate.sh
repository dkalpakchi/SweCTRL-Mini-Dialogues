#!/bin/bash
CODES=("forum" "forum/economy" "forum/law" "forum/sport" "forum/tech" "forum/travel" "news" "debate")
NPF="-npf"

for code in "${CODES[@]}"
do
  python3.8 generate.py -f ~/swectrl/ctrl_minip_ddp_256_v4/checkpoint-2786844 -caf ~/swectrl/ctrl_minip_ddp_256_v4/ctrl_args.bin -c ${code} -l 4096 -n 20 $NPF
  python3.8 generate.py -f ~/swectrl/ctrl_minip_ddp_256_v4/checkpoint-2786844 -caf ~/swectrl/ctrl_minip_ddp_256_v4/ctrl_args.bin -c ${code} -l 4096 -gd $NPF
done
