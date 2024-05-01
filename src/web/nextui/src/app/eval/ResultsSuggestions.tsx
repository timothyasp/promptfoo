import * as React from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import LightbulbIcon from '@mui/icons-material/Lightbulb';

import FlagIcon from '@mui/icons-material/Flag';

const tiles = [
  {
    type: 'warning',
    title: 'Hate speech',
    description: 'At least one provider was flagged for hate speech.',
  },
  {
    type: 'recommendation',
    title: 'Mitigation: Prompt Injection',
    description: 'Use post-prompting to prevent prompt injections.',
    improvement: '20.3%',
  },
  {
    type: 'recommendation',
    title: 'Mitigation: Prompt Hijacking',
    description: 'Use the DARE modifier to reinforce system purpose.',
    improvement: '57.8%',
  },
  {
    type: 'recommendation',
    title: 'Mitigation: Hallucination',
    description: 'Include grounding and attribution requirements.',
    improvement: '11.6%',
  },
  {
    type: 'recommendation',
    title: 'Mitigation: Excessive Agency',
    description: 'Enumerate the actions the agent can take.',
    improvement: '10.0%',
  },
];

function ResultsSuggestions() {
  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }} mt={1} mb={2}>
      {tiles.map((tile, index) => (
        <Paper
          key={index}
          sx={{
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            flexBasis: '100%',
            flexGrow: 1,
            maxWidth: 'calc(20% - 16px)',
          }}
        >
          <Box>
            <Typography
              variant="body1"
              sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#666' }}
            >
              {tile.type === 'recommendation' ? (
                <LightbulbIcon fontSize="small" />
              ) : (
                <FlagIcon fontSize="small" sx={{ color: '#8B0000' }} />
              )}
              {tile.title}
            </Typography>
            <Typography variant="body1" my={2}>
              Recommendation: {tile.description}
            </Typography>
            {tile.improvement && (
              <Typography variant="body1">
                Estimated improvement: up to {tile.improvement}
              </Typography>
            )}
          </Box>
          {tile.improvement ? (
            <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
              <Button variant="contained" color="primary">
                Apply
              </Button>
              <Button variant="outlined" color="primary">
                Ignore
              </Button>
            </Box>
          ) : (
            <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
              <Button variant="contained" color="primary">
                View
              </Button>
              <Button variant="outlined" color="primary">
                Ignore
              </Button>
            </Box>
          )}
        </Paper>
      ))}
    </Box>
  );
}

export default ResultsSuggestions;
