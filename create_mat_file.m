function create_mat_file(dir, output)

%Generates .mat file for Automated CMB Detection
%Written by Yicheng Chen 2019
%Modified by Melanie Morrison 2022

%dir = path to subject subfolders, where you will also store the output of subjectlist.m
%output = output path, likely samepath as dir

clear; clc;
start_i = 1;

% --------------- load datadir.mat file ----------------
load([dir '/datadir.mat']) % run subjectlist.m to generate datadir.mat first.
mkdir(output)

% Extract required data from images and labels
% save as .mat data files. Name: {Subject_Num}_{Scan_Num}.mat
for i = start_i : start_i + length(datadir)
    
    scans = datadir{i - start_i + 1};
    
    for j = 1:length(scans)
        
        try
        
            fprintf('Subject %d, Scan %d\n', i, j);

            [f1, fp_file] = system(['ls ' scans{i} '/*nonproj*false_positives*.nii']);
            [f2, cmb_file] = system(['ls ' scans{j} '/*nonproj*finalcorrected*.nii']);
            [f3, swi_file] = system(['ls ' scans{j} '/*scaled.nii']);
            

            fp_file = strsplit(fp_file, '\n');
            fp_file = [fp_file{1, 1} ' '];
            
            cmb_file = strsplit(cmb_file, '\n');
            cmb_file = [cmb_file{1, 1} ' '];
            
            swi_file = strsplit(swi_file, '\n');
            swi_file = [swi_file{1, 1} ' '];
            
            fprintf('%s\n%s\n%s\n', swi_file, fp_file, cmb_file);

            cmb = load_untouch_nii(cmb_file(1:end-5));
            fp = load_untouch_nii(fp_file(1:end-5));
            swi = load_untouch_nii(swi_file(1:end-5));

            cmb_mask = cmb.img;
            fp_mask = fp.img;

            pixel_size = cmb.idf.pixelsize;
            disp(pixel_size);

            centroids_cmb = mask2centroid(cmb_mask);
            centroids_fp = mask2centroid(fp_mask);

            swi = swi.img;

            n_cmb = size(centroids_cmb, 1);
            n_fp = size(centroids_fp, 1);
            scan_name = datadir{i-start_i+1}{j};

            save([output_dir num2str(i-1) '_' num2str(j) '.mat'], ...
                'pixel_size', 'centroids_cmb', 'centroids_fp', 'swi', 'cmb_mask', 'fp_mask', ...
                'n_cmb', 'n_fp', 'scan_name', 'cmb_file', 'fp_file', 'swi_file');
        
        catch
            disp('Error!')
            continue
        end
        % -----------------------------------------------------
        % Each mat contains data of one scan:
        %   scan_name:          name of the scan
        %   cmb_file:           CMB label(mask) filename
        %   fp_file:            FP label(mask) filename
        %   swi_file:           SWI filename
        %   pixel_size:         Pixel size of the scan
        %   centroids_cmb:      centroids of CMBs, n_cmbx3 matrix
        %   centroids_fp:       centroids of FPs, n_fpx3 matrix
        %   swi:                SWI volume
        %   cmb_mask:           CMB binary mask
        %   fp_mask:            FP binary mask
        %   n_cmb:              number of CMBs
        %   n_fp:               number of FPs
        
    end
end


%% check saved mat and get a summary (save as txt file)

total_cmb = 0;
total_fp = 0;


summary_file = fullfile(output, 'data_summary.txt');
fid = fopen(summary_file,'w');

for i = 1:length(datadir)
    patient = datadir{i};
    for j = 1:length(patient)
        try
            load([output num2str(i-1) '_' num2str(j) '.mat'])
            line = sprintf('%3d-%2d: Scan name: %60s, CMBs: %4d, FPs: %4d\n', i-1, j, [dir scan_name], n_cmb, n_fp);
            fprintf(fid, line);
            fprintf(line);

            total_cmb = total_cmb + n_cmb;
            total_fp = total_fp + n_fp;
        catch
            continue
        end
    end
end

line = sprintf('Total CMB: %d\nTotal FP : %d\n', total_cmb, total_fp);
fprintf(fid, line);
fprintf(line);

fprintf('Wrote to file: %s\n', summary_file);
