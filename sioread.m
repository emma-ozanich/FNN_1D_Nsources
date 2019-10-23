function x=sioread(filename,p1,npi,channels);
%-----------------------------------------------------------------------
% sioread.m
%
% This program runs under windows, unix, and macs.  
%
% function x=sioread(filename,p1,npi,channels);
%
% Inputs:
% 	filename: Name of sio file to read
% 	p1:	Point to start reading ( 0 < p1 < np)
% 	npi: 	Number of points to read in
% 	channels: Single number or vector containing the channels to read
% 		(example-to read channels 4 thru 10 enter 4:10)
%
% Example:  xdata = sioread('../data.dir/test1.sio',10,100,[1 2 4:6]);
% xdata will be a matlab array of 100 points and 5 channels.  The
% points start at the siofile point #10 and the channels extracted are
% channels 1,2,4,5,6
%
%
% originally written by Aaron Thode.  Modified by Geoff Edelmann and 
% James Murray. (Final Version: July, 2001)
%-----------------------------------------------------------------------
 
% pathname=[pwd '/' filename];

% since the majority of sio files will be created as 'SUN' sio
% files, we will assume they are 'big endian' and check for
% compliance.
%
endian='b';
fid=fopen(filename,'r',endian);
fseek(fid,28,-1);
bs=fread(fid,1,'long');  % should be 32677
if bs ~= 32677
  fclose(fid);
  endian='l';
  fid=fopen(filename,'r',endian);
  fseek(fid,28,-1);
  bs=fread(fid,1,'long'); % should be 32677
  if bs ~= 32677
    error(['Problem with byte swap constant:' num2str(bs)])
  end
end

% I can just use fseek(fid,0,-1) to position to beginning of file
% but I think that closing and reopening is cleaner.
fclose(fid);
fid=fopen(filename,'r',endian);
 

 id=fread(fid,1,'long');  % ID Number
 nr=fread(fid,1,'long');  % Number of Records in File
 rl=fread(fid,1,'long');  % Record Length in Bytes
 nc=fread(fid,1,'long');  % Number of Channels in File
 sl=fread(fid,1,'long');  % Bytes/ Point
 type = 'float';
 if sl==2,
        type='short';
 end
 f0=fread(fid,1,'long');  % 0 = integer, 1 = real
 np=fread(fid,1,'long');  % Number of Points per Channel
 bs=fread(fid,1,'long');  % should be 32677
 fn=fread(fid,24,'char'); % Read in the file name
 com= fread(fid,72,'char'); % Read in the Comment String
 
 rechan=ceil(nr/nc);     %records per channel
 ptrec=rl/sl;            %points/record

% check to make sure the channel list is valid
for ii = 1:length(channels)
  if (channels(ii) <= 0) 
    error(['Channel must be positive: ' channels(ii)])
  end
  if (channels(ii) > nc) 
    error(['Channel does not exist:' channels(ii)])
  end
end

%-- First calculate what time period is desired

 r1=floor(p1/ptrec)+1;		%record where we wish to start recording
 if npi <= 0			%if npi is 0, then we want all points		
   npi = np;
 end
 p2=p1+npi-1;			%ending position in points for each channel.
 r2=ceil(p2/ptrec);		%record where we end recording

 if p2>np,			%-- Error checking to prevent reading past EOF
       disp('Warning:  p1 + npi > np');
       p2=np;r2=rechan;
 end
 totalrec=r2-r1+1;		%number of records desired read.

% we're going to read in entire records into the variable tmpx, then
% we'll worry about the start point.

 pp1 = mod(p1,ptrec);			% p1's position in record
 if pp1 == 0				% no remainder means last point in record
   pp1 = 2048;
 end
 x = zeros(p2-p1+1,length(channels));	% "allocate" a matrix for final result
 tmpx = zeros(totalrec*ptrec,1);	% make a temporary matrix
% Loop over the desired channels
 for J=1:length(channels)
    count = 1;				% Start 
    trec = (r1-1)*nc + channels(J);	% First Record to read for this channel
    for R = 1:totalrec
      status = fseek(fid,rl*trec,-1);        % position to the desired record
      if status == -1
        error(ferror(fid))
      end
      tmpx(count:count+ptrec-1) = fread(fid,ptrec,type);	% Read in a record's worth of points
      count = count+ptrec;	% adjust for the next set of points
      trec = trec + nc;		% Next record for this channel is nc records away
    end
    x(1:p2-p1+1,J) = tmpx(pp1:pp1+p2-p1);
 end


 fclose(fid);

%-- end of program
