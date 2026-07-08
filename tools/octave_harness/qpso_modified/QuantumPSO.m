function res = QuantumPSO(cost_fn, PSOparams, D, mv, VR, minmax, PSOseedValue)
  % HARNESS MODIFIED-COPY of legacy/Optimizer_V6.2.2/functions/QuantumPSO.m (NOT an edit to
  % legacy/). The vendored source is bit-pinned by driving THIS copy over a shared random
  % table (tools/octave_harness/qpso_modified/qpso_random via qpso_reset_cursor) and comparing
  % against the Python port (speech.optimizers.quantum_pso).
  %
  % Every substitution is a % HARNESS-SUB comment quoting the original line. Summary:
  %   - QuantumPSO.m:91  `rand('state',sum(100*clock));`  -> REMOVED (the wall-clock reseed;
  %                       replaced by the injected shared table via qpso_reset_cursor).
  %   - every `rand(...)`      -> `tbl_rand(...)`      (column-major table read).
  %   - `randperm(...)`        -> `tbl_randperm(...)`  (table Fisher-Yates).
  %   - `stblrnd(1.3,1,0.5,0,1,D)` -> `tbl_stblrnd(...)` (table Chambers-Mallows-Stuck).
  %   - every `feval(functname, pos, PS, mode)` cost call -> `cost_fn(pos, mode)` (the
  %                       quadratic surrogate qpso_surrogate; the engine feval is unreachable
  %                       without a real corpus).
  %   - the nargin dispatch (:89-174), plotting/LogAndDisplay, the validation cadence
  %                       (:702-726), and the RpropFinal/backprop refinement are dropped: the
  %                       caller passes fixed args, PS.NS.BackPropagationActivated == 0 so the
  %                       :481 `&&` short-circuits (NO draw), and errgoal == NaN disables the
  %                       :766 goal block. The DEAD velocity banks (:375-411) are KEPT VERBATIM
  %                       (they draw the stream even though the apply gate :447 is unreachable).
  %
  % Returns a struct: init_* (post-init state, :327), the per-epoch *_traj snapshots, OUT
  % (:841), te, and cursor_end (the shared-cursor position for the draw-count guard).

  df      = PSOparams(1);   %#ok<NASGU>
  me      = PSOparams(2);
  ps      = PSOparams(3);
  ac1     = PSOparams(4);
  ac2     = PSOparams(5);
  iw1     = PSOparams(6);
  iw2     = PSOparams(7);
  iwe     = PSOparams(8);
  ergrd   = PSOparams(9);
  ergrdep = PSOparams(10);
  errgoal = PSOparams(11); %#ok<NASGU>  % NaN -> the :766 goal block is dropped
  trelea  = PSOparams(12);
  PSOseed = PSOparams(13);

  % INITIALIZE INITIALIZE INITIALIZE (:257)
  pos = normmat(tbl_rand([2*ps,D]), VR', 1);   % HARNESS-SUB rand->tbl_rand | QuantumPSO.m:257 normmat(rand([2*ps,D]),VR',1)
  if PSOseed == 1                              % :259
    tmpsz = size(PSOseedValue);
    pos(1:tmpsz(1),1:tmpsz(2)) = PSOseedValue; % :261
  end
  % construct initial random velocities (:265-272) -- DEAD (vels never applied) but drawn
  vel1 = normmat(tbl_rand([ps,D]), [forcecol(-mv),forcecol(mv)]',1);  % HARNESS-SUB | :265-266
  vel2 = normmat(tbl_rand([ps,D]), [forcecol(-mv),forcecol(mv)]',1);  % HARNESS-SUB | :267-268
  vel3 = normmat(tbl_rand([ps,D]), [forcecol(-mv),forcecol(mv)]',1);  % HARNESS-SUB | :269-270
  vel4 = normmat(tbl_rand([ps,D]), [forcecol(-mv),forcecol(mv)]',1);  % HARNESS-SUB | :271-272

  gbestval = 1e32;                             %#ok<NASGU>  % :277
  [out, pos_out] = cost_fn(pos, -2);           % HARNESS-SUB feval->cost_fn | :279
  % Opposition Based Learning (:282-288)
  minbound = min(pos_out,[],1);                % :282
  maxbound = max(pos_out,[],1);                % :283
  pos_out_opp = repmat(minbound+maxbound,size(pos_out,1),1)-pos_out;  % :284
  [out_opp, pos_out_opp] = cost_fn(pos_out_opp, -2);  % HARNESS-SUB | :285
  out = [out; out_opp];                        % :287
  pos_out = [pos_out; pos_out_opp];            % :288
  [iterbestval, idx1] = min(out);              % :290
  gbestval = iterbestval;                      % :291
  gbest = pos_out(idx1,:);                     % :292
  GApos = []; outGA = [];                      % :297-299
  [out, pos_out] = cost_fn(pos_out, -1);       % HARNESS-SUB | :301
  tmp_sort = sortrows([out (1:numel(out))']);  % :303
  pos = pos_out(tmp_sort(1:ps,2),:);           % :309
  out = out(tmp_sort(1:ps,2),:);               % :310
  pbest = pos;                                 % :314
  pbestval = out;                              % :315
  xreal = [gbest;pbest;GApos]';                % :319
  cout = [gbestval;pbestval;outGA]';           % :320
  [iterbestval, idx1] = min(cout);             % :322
  if gbestval >= iterbestval                   % :323
    gbestval = iterbestval;                    % :324
    gbest = xreal(:,idx1)';                    % :325
  end

  res.init_pos = pos;
  res.init_pbest = pbest;
  res.init_pbestval = pbestval;
  res.init_gbest = gbest;
  res.init_gbestval = gbestval;

  if (trelea == 3)                             % :333 (Clerc constriction; dead vel3 uses chi)
    kappa = 1;
    if ((ac1+ac2) <= 4)
      chi = kappa;
    else
      psi = ac1 + ac2;
      chi_den = abs(2-psi-sqrt(psi^2 - 4*psi));
      chi_num = 2*kappa;
      chi = chi_num/chi_den;
    end
  end

  tr = ones(1,me)*NaN;                         % :233
  cnt2 = 0;                                    % :350
  te = 0;
  iwt = zeros(1,me);
  pos_traj = []; pbest_traj = []; pbestval_traj = [];
  gbest_traj = []; gbestval_traj = [];

  for i = 1:me                                 % :364
    % --- DEAD velocity banks (:375-411): drawn, never applied (gate :447 unreachable) ---
    rannum1 = tbl_rand([ps,D]); rannum2 = tbl_rand([ps,D]);  % HARNESS-SUB | :375-376
    vel1 = 0.729.*vel1 + 1.494.*rannum1.*(pbest-pos) + 1.494.*rannum2.*(repmat(gbest,size(pos,1),1)-pos);  % :378-380
    rannum1 = tbl_rand([ps,D]); rannum2 = tbl_rand([ps,D]);  % HARNESS-SUB | :382-383
    vel2 = 0.600.*vel2 + 1.700.*rannum1.*(pbest-pos) + 1.700.*rannum2.*(repmat(gbest,size(pos,1),1)-pos);  % :385-387
    rannum1 = tbl_rand([ps,D]); rannum2 = tbl_rand([ps,D]);  % HARNESS-SUB | :389-390
    vel3 = chi*(vel3 + ac1.*rannum1.*(pbest-pos) + ac2.*rannum2.*(repmat(gbest,size(pos,1),1)-pos));  % :392-394
    if i<=iwe                                  % :398
      iwt(i) = ((iw2-iw1)/(iwe-1))*(i-1)+iw1;  % :399
    else
      iwt(i) = iw2;                            % :401
    end
    rannum1 = tbl_rand([ps,D]); rannum2 = tbl_rand([ps,D]);  % HARNESS-SUB | :403-404
    ac11 = rannum1.*ac1; ac22 = rannum2.*ac2;  % :406-407
    vel4 = iwt(i).*vel4 + ac11.*(pbest-pos) + ac22.*(repmat(gbest,size(pos,1),1)-pos);  % :409-411

    % --- QDPSO update (:414-443) ---
    coefExpContr = 0.75-0.5*(i-1)/(me-1);      % :415
    phi = tbl_rand(ps,D);                      % HARNESS-SUB | :416
    u = max(1e-32, tbl_rand(ps,D));            % HARNESS-SUB | :417
    MBest = repmat(mean(pbest,1),size(pos,1),1); %#ok<NASGU>  % :418 (dead; kept faithful)
    attractor = phi.*pbest+(1-phi).*repmat(gbest,size(pos,1),1);  % :419
    signs = sign(tbl_rand(ps,D)-0.5);          %#ok<NASGU>  % HARNESS-SUB | :420 DEAD (overwritten :442)

    % Ranking Operator variant (:424-440)
    localAttractor = zeros(size(attractor));   % :424
    [rank,irank] = sortrows(pbestval);         % :425
    elem = [irank (ps:-1:1)'];                 % :426
    for ii = 1:ps                              % :427
      rank = elem(:,2).*(elem(:,2) > ii);      % :428
      sector = rank/sum(rank);                 % :429
      sect_cs = cumsum(sector);                % :430
      wheel = tbl_rand(1,1);                   % HARNESS-SUB | :432
      tmp = find(sect_cs >= wheel);            % :433
      if isempty(tmp)                          % :434
        localAttractor(ps+1-ii,:) = gbest;     % :435
      else
        localAttractor(ps+1-ii,:) = pbest(irank(tmp(1)),:);  % :438
      end
    end
    attractor = phi.*pbest+(1-phi).*localAttractor;  % :441
    signs = sign(tbl_rand(ps,D)-0.5);          % HARNESS-SUB | :442
    pos = attractor+coefExpContr*signs.*max(1e-5,abs(attractor-pos)).*log(1./u);  % :443

    % --- update new position (:447) is DEAD: `i > 20*me/2` never true; no draws ---

    % --- DE recombination + Levy (:463-479) ---
    xreal_new = [];                            % :463
    indexes = [];                              % :464
    for ii = 1:size(pos,1)                     % :465
      if (tbl_rand() < 0.2)                    % HARNESS-SUB | :466
        neighboors = tbl_randperm(size(pos,1),2);  % HARNESS-SUB | :467
        ponderations = tbl_rand(1,3);          % HARNESS-SUB | :468
        ponderations = ponderations/sum(ponderations);  % :469
        new_pos = ponderations(1)*pos(ii,:)+ponderations(2)*(pbest(ii,:)-pos(ii,:))+ponderations(3)*(pos(neighboors(1),:)-pos(neighboors(2),:));  % :470
        xreal_new = [xreal_new;new_pos];       % :471
        indexes = [indexes ii];                % :472
      end
      if (tbl_rand() < 0.2)                    % HARNESS-SUB | :474
        xreal_new = [xreal_new;pos(ii,:)+tbl_stblrnd(1.3,1,0.5,0,1,D).*sign(tbl_rand(1,D)-0.5)];  % HARNESS-SUB | :475
        indexes = [indexes ii];               % :477
      end
    end

    % --- backprop refine gate (:480-485): HOOK. BackPropagationActivated == 0 in the
    %     surrogate, so the `&&` short-circuits and NO `rand < 0.5` draw is consumed. ---

    % --- cost eval (:563) ---
    [out, pos_out] = cost_fn([pos; xreal_new], i);  % HARNESS-SUB | :563
    xreal_tmp = pos_out(ps+1:end,:);           % :595
    cout_tmp = out(ps+1:end);                  % :596
    pos = pos_out(1:ps,:);                     % :597
    out = out(1:ps,:);                         % :598
    for ii = 1:length(indexes)                 % :600
      if (cout_tmp(ii) < out(indexes(ii)))     % :601
        pos(indexes(ii),:) = xreal_tmp(ii,:);  % :603
        out(indexes(ii)) = cout_tmp(ii);       % :604
      end
    end
    xreal = [gbest;pbest;pos;GApos]';          % :623
    cout = [gbestval;pbestval;out;outGA]';     % :624
    [iterbestval,idx1] = min(cout);            % :626
    if gbestval >= iterbestval                 % :627
      gbestval = iterbestval;                  % :628
      gbest = xreal(:,idx1)';                  % :629
    end
    tr(i+1) = gbestval;                        % :633
    te = i;                                    % :634
    if minmax == 0                             % :652
      tempi = find(pbestval>=out);             % :653
      pbestval(tempi,1) = out(tempi);          % :654
      pbest(tempi,:) = pos(tempi,:);           % :655
    end

    pos_traj = [pos_traj pos];                 % snapshot: ps x (D*me)
    pbest_traj = [pbest_traj pbest];
    pbestval_traj = [pbestval_traj pbestval];
    gbest_traj = [gbest_traj; gbest];          % me x D
    gbestval_traj = [gbestval_traj gbestval];  % 1 x me

    % --- stall termination (:751-763); errgoal=NaN disables the :766 goal block ---
    tmp1 = abs(tr(i) - gbestval);              % :751
    if tmp1 > ergrd                            % :752
      cnt2 = 0;
    elseif tmp1 <= ergrd                       % :754
      cnt2 = cnt2+1;
      if cnt2 >= ergrdep
        break                                  % :761
      end
    end
  end                                          % :795

  % --- FINAL GENERATION (:799-843) ---
  [outfinal, pos_out] = cost_fn([gbest;pbest;GApos], 0);  % HARNESS-SUB | :808
  gbest = pos_out(1,:);                        % :812
  gbestval = outfinal(1,:);                    % :813
  pbest = pos_out(2:size(pbest,1)+1,:);        % :815
  pbestval = outfinal(2:size(pbest,1)+1,:);    % :816
  xreal = [gbest;pbest;GApos]';                % :821
  cout = [gbestval;pbestval;outGA]';           % :822
  [iterbestval,idx1] = min(cout);              % :824
  if gbestval >= iterbestval                   % :832
    gbestval = iterbestval;                    % :833
    gbest = xreal(:,idx1)';                    % :834
  end
  OUT = [gbest';gbestval];                     % :841

  global QPSO_CURSOR
  res.pos_traj = pos_traj;
  res.pbest_traj = pbest_traj;
  res.pbestval_traj = pbestval_traj;
  res.gbest_traj = gbest_traj;
  res.gbestval_traj = gbestval_traj;
  res.OUT = OUT;
  res.te = te;
  res.cursor_end = QPSO_CURSOR - 1;   % 0-based draw count consumed (Python TableRng.cursor)
end
