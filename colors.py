"""
Liste de colours -- a enrichir

Pour tester :
c = 31
for i in range(8):
    sys.stdout.write("\033["+str(i)+";"+str(c)+"m")
    print("Format TEST : ", str(i)+";"+str(c)+"m")

"""

RED			= "\033[0;31m"  
DARKRED		= "\033[1;31m"  
REV_RED		= "\033[7;31m"
REV_RED2	= "\033[1;41m"
GREEN		= "\033[0;32m"
DARKGREEN	= "\033[1;32m"
REV_GREEN	= "\033[7;32m"
REV_GREEN2	= "\033[1;42m"
ORANGE		= "\033[0;33m"  
DARKORANGE	= "\033[1;33m"  
REV_ORANGE	= "\033[7;33m"
REV_ORANGE2	= "\033[1;43m"
BLUE		= "\033[0;34m"
DARKBLUE	= "\033[1;34m"  
REV_BLUE	= "\033[7;34m"
REV_BLUE2	= "\033[1;44m"


PINK		= "\033[1;35m"
CYAN		= "\033[1;36m"
GREY		= "\033[1;37m"
BLACK		= "\033[1;38m"

RESET 		= "\033[0;0m"
BOLD    	= "\033[;1m"
REVERSE 	= "\033[;7m"


