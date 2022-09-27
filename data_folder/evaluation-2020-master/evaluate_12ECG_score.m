% This file contains functions for evaluating algorithms for the 2020 PhysioNet/
% Computing in Cardiology Challenge. You can run it as follows:
%
%   evaluate_12ECG_score(labels, outputs, scores.csv)
%
% where 'labels' is a directory containing files with the labels, 'outputs' is a
% directory containing files with the outputs from your model, and 'scores.csv'
% (optional) is a collection of scores for the algorithm outputs.
%
% Each file of labels or outputs must have the format described on the Challenge
% webpage. The scores for the algorithm outputs include the area under the
% receiver-operating characteristic curve (AUROC), the area under the recall-
% precision curve (AUPRC), accuracy (fraction of correct recordings), macro F-
% measure, and the Challenge metric, which assigns different weights to
% different misclassification errors.

function evaluate_12ECG_score(labels, outputs, output_file, class_output_file)
    % Check for Python and NumPy.
    command = 'python -V';
    [status, ~] = system(command);
    if status~=0
        error('Python not found: please install Python or make it available by running "python ...".');
    end

    command = 'python -c "import numpy"';
    [status, ~] = system(command);
    if status~=0
        error('NumPy not found: please install NumPy or make it available to Python.');
    end

    % Define command for evaluating model outputs.
    switch nargin
        case 2
            command = ['python evaluate_12ECG_score.py' ' ' labels ' ' outputs];
        case 3
            command = ['python evaluate_12ECG_score.py' ' ' labels ' ' outputs ' ' output_file];
        case 4
            command = ['python evaluate_12ECG_score.py' ' ' labels ' ' outputs ' ' output_file ' ' class_output_file];
        otherwise
            command = '';
    end

    % Evaluate model outputs.
    [~, output] = system(command);
    fprintf(output);
end
