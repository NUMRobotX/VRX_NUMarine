savedcmd_/home/nuc1/Downloads/linuxcan/virtualcan/kvvirtualcan.mod := printf '%s\n'   virtualcan.o | awk '!x[$$0]++ { print("/home/nuc1/Downloads/linuxcan/virtualcan/"$$0) }' > /home/nuc1/Downloads/linuxcan/virtualcan/kvvirtualcan.mod