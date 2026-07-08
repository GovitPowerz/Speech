function [indBest,indMiddle,indWorst,scoreBest,scoreMiddle,scoreWorst] = getworstandbest_transcribed(file_nb,sortingTable,nbWorst,nbBest,nbMiddle)
  % legacy: Optimizer_V6.2.2/functions/getCases.m:116-165 (getWorstAndBest), transcribed
  % VERBATIM (no logic changes, only the function name/file to make it freestanding).
  %
  % FALLBACK TIER (same precedent as stage_computecost.m -- see IMPROVEMENTS.md): this is a
  % MATLAB subfunction PRIVATE to getCases.m (a function FILE exposes only its FIRST
  % function externally; later `function` defs in the same file are file-local helpers),
  % and getCases.m itself is unreachable standalone -- its first line shells out through
  % the full ComputeCost/CostFunction engine chain (getCases.m:5). Driven directly by
  % stage_batching.m over crafted sortingTable inputs.

  indWorst = [];
  scoreWorst = [];
  if (nbWorst > 0)
      count = file_nb;
      while ((length(indWorst) < nbWorst)&&(count > 0))
          if ((length(indWorst) == 0)||(sum(indWorst == sortingTable(count,2)) == 0))
              indWorst = [indWorst;sortingTable(count,2)];
              scoreWorst = [scoreWorst;sortingTable(count,1)];
          end
          count = count-1;
      end
      nbWorst = count+1;
  else
      nbWorst = file_nb;
  end
  indWorst = flipud(indWorst);
  scoreWorst = flipud(scoreWorst);

  indBest = [];
  scoreBest = [];
  if (nbBest > 0)
      count = 1;
      while ((length(indBest) < nbBest)&&(count <= file_nb))
          if (((length(indBest) == 0)&&(sum(indWorst == sortingTable(count,2)) == 0))||((sum(indBest == sortingTable(count,2)) == 0)&&(sum(indWorst == sortingTable(count,2)) == 0)))
              indBest = [indBest;sortingTable(count,2)];
              scoreBest = [scoreBest;sortingTable(count,1)];
          end
          count = count+1;
      end
      nbBest = count-1;
  else
      nbBest = 1;
  end

  indMiddle = [];
  scoreMiddle = [];
  if (nbMiddle > 0)
      step = max(1,round((nbWorst-nbBest+1)/(nbMiddle+1)));
      count = nbBest+step;
      while ((length(indMiddle) < nbMiddle)&&(count <= nbWorst))
          index = sortingTable(count,2);
          if (((length(indMiddle) == 0)&&(sum(indBest == index) == 0)&&(sum(indWorst == index) == 0))||((sum(indMiddle == index) == 0)&&(sum(indBest == index) == 0)&&(sum(indWorst == index) == 0)))
              indMiddle = [indMiddle;index];
              scoreMiddle = [scoreMiddle;sortingTable(count,1)];
          end
          count = count+step;
      end
  end
end
