                     

                                                                           
                                        

from GA_models.src import config


def shouldTerminate(population, gen, duration):
    
    exit = False
    if gen > config.maxGen:
        exit=True
    elif duration > config.maxDuration:
        exit = True
    
    return exit
