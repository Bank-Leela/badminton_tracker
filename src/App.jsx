import React, { useState } from 'react';
import { Trophy, Calendar, User, ChevronRight, PlusCircle } from 'lucide-react';

const BadmintonTracker = () => {
  const [matches, setMatches] = useState([
    { id: 1, opponent: 'Alex Chen', myScore: 21, oppScore: 18, date: 'Jan 15, 2026', location: 'CIF' },
    { id: 2, opponent: 'Jordan Lee', myScore: 19, oppScore: 21, date: 'Jan 17, 2026', location: 'PAC' },
  ]);

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 p-6 font-sans">
      <div className="max-w-4xl mx-auto">
        
        {/* Header Area */}
        <header className="flex justify-between items-end mb-10">
          <div>
            <h1 className="text-4xl font-extrabold text-white tracking-tight">CourtSide</h1>
            <p className="text-slate-400 mt-1">UWaterloo Season Stats</p>
          </div>
          <button className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg transition-all font-semibold shadow-lg shadow-blue-900/20">
            <PlusCircle size={20} /> New Match
          </button>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* Quick Stats Summary */}
          <div className="md:col-span-1 space-y-4">
            <div className="bg-slate-800/50 border border-white/10 p-6 rounded-2xl backdrop-blur-sm">
              <p className="text-slate-400 text-sm uppercase tracking-wider font-semibold">Win Rate</p>
              <h2 className="text-3xl font-bold text-blue-400 mt-1">67%</h2>
            </div>
            <div className="bg-slate-800/50 border border-white/10 p-6 rounded-2xl backdrop-blur-sm">
              <p className="text-slate-400 text-sm uppercase tracking-wider font-semibold">Total Games</p>
              <h2 className="text-3xl font-bold text-white mt-1">12</h2>
            </div>
          </div>

          {/* Match History List */}
          <div className="md:col-span-2 space-y-4">
            <h3 className="text-xl font-bold text-white flex items-center gap-2 mb-4">
              <Calendar size={20} className="text-blue-500" /> Recent Matches
            </h3>
            
            {matches.map((match) => (
              <div 
                key={match.id} 
                className="group bg-slate-800/30 border border-white/5 hover:border-blue-500/50 p-5 rounded-2xl transition-all flex items-center justify-between"
              >
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-full ${match.myScore > match.oppScore ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
                    <Trophy size={24} />
                  </div>
                  <div>
                    <h4 className="font-bold text-white group-hover:text-blue-400 transition-colors">vs. {match.opponent}</h4>
                    <p className="text-sm text-slate-500">{match.date} • {match.location}</p>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  <div className="text-right">
                    <span className={`text-2xl font-black ${match.myScore > match.oppScore ? 'text-green-400' : 'text-slate-400'}`}>
                      {match.myScore}
                    </span>
                    <span className="mx-2 text-slate-600 font-bold">-</span>
                    <span className={`text-2xl font-black ${match.oppScore > match.myScore ? 'text-green-400' : 'text-slate-400'}`}>
                      {match.oppScore}
                    </span>
                  </div>
                  <ChevronRight size={20} className="text-slate-600 group-hover:text-white transition-all" />
                </div>
              </div>
            ))}
          </div>

        </div>
      </div>
    </div>
  );
};

export default BadmintonTracker;