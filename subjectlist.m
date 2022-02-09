datadir = {
{'/working/lupolab/cmb_detection_2018_nifti/test_subject/PD001'};
{'/working/lupolab/cmb_detection_2018_nifti/test_subject/PD002'};
%{'P003'};
%{'P004'};
};

for i = 1:length(datadir)
    n_scan(i) = size(datadir{i}, 2);
end

save('/YOUR/OUTPUT/PATH/FOT/datadir.mat', ...
    'datadir', 'n_scan');