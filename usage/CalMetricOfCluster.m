function [acc_r nmi_r] = CalMetricOfCluster(Predict_label,ttls)
if size(Predict_label,1)<size(Predict_label,2)
    Predict_label=Predict_label';
end;
if size(ttls,1)<size(ttls,2)
    ttls=ttls';
end;
for i = 1:size(Predict_label,2)
    acc_r(i) = accuracy(ttls, Predict_label(:,i));
    nmi_r(i) = nmi(ttls, Predict_label(:,i));
end


