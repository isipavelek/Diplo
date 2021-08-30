import sys
from preprocessing import general_analysis, retargeting_analysis
from utils import load_dataframe, set_project_folders


if __name__ == '__main__': 
    
    print('Seting projects outputs folders \r')
    project_dirs = set_project_folders()
    
    # Load DataFrame
    print('Loading DataSet --> {:s}\r'.format(sys.argv[1]))
    df = load_dataframe(sys.argv[1])  
    print('{:s} --> successfully loaded \r'.format(sys.argv[1]))

    # Explore, transform and analysis input DataFrame
    print('Starting general analysis \r')
    df_clean = general_analysis(input_dataframe=df, output_dirname=project_dirs[0])                                                                                                   

    # Perform retargeting analysis
    print('Performed retargeting/remarketing analysis with clean DataSet \r')
    retargeting_analysis(input_dataframe=df_clean, output_dirname=project_dirs[1::])  

    print('See results on ./results folder \r')                                                                      



