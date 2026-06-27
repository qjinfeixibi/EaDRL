import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")
    parser.add_argument('--algorithm', type=str, default="gt_drl", help="one of matnet, hgnn, attention, matnet_attention")
    parser.add_argument('--cuda', type=int, default=0, help="cuda device number")
    parser.add_argument('--log_file_desc', type=str, default='heteronet_train', help='file name of results to store')
    parser.add_argument('--enable_change_paras', action='store_true', help='train the model over the changing environment')
    parser.add_argument('--test_seed', default=6, type=int, help='test seed')
    parser.add_argument('--CPoptimizer', action='store_true')
    parser.add_argument('--static', action='store_true', help='generate env parameters with static method')
    parser.add_argument('--metarl', action='store_true', help='use metarl training')
    parser.add_argument('--metarl_subgraphs', action='store_true', help='use metarl training')
    parser.add_argument('--subprob', action='store_true', help='use metarl training')
    parser.add_argument('--multi_test', action='store_true', help='run multiple tests')
    parser.add_argument('--job_centric', action='store_true', help='use job instead of ope')
    parser.add_argument('--test_GA', action='store_true', help='run GA algorithm tests')
    parser.add_argument('--test_dispatch', action='store_true', help='run dispatch rule tests')
    parser.add_argument('--benchmark_file', type=str, default='', help='benchpark dataset file name')
    parser.add_argument('--new_job', action='store_true', help='new job insertion')

                 
                                                                                        
                                                                                                                                        
                                                                                          
                                                                                           
                                                                                                    
                                                                                                 
    
             
                                                                                                      
                                                                         
                                                                                                        
                                                                          
                                                                            
                                                                         
                                                                            
                                                                           
                                                                                
                                                                       
    
    
    
            
                                                                                                 
                                                                                                     
                                                                                                                          
                                                                
                                                                                               
                                                                                                             

             
                                                                                                           
                                                                                                        
                                                                                                              
                                                                   
                                                                                
                                                                     
                                                                                         
                                                                       
                                                                                                                         

                
                                                                                                                     
                                                                                                                       
                                                                                                      
                                                                                                           
                                                                                                    
                                                                                      
                                                                     
                                                                                                            
                                                                                
                                                                
                                                                                         
                                                     
                                                                                                                 
                                                                 
                                                                                          
                                                                       
                                                                                                                        
                                                                                                              
                                                                      
                                                                                
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
                                                                  
                                                                                                                
                                                                                   
                                                                        
                                                                                                                   

          
                                                                                                   
                                                                                                            
                                                                                       
                                                                                                        
                                                               
                                                                                     
                                                                     
                                                                                                      
                                                                                                       
                                                                                  
                                                                                                            
                                                                                                

    opts = parser.parse_args(args)

                                                                    
                                                                                   
                                   
                          
                                                        
                       
       
                                       
                                                                        
                                                                         
                                                                                                         
    return opts
