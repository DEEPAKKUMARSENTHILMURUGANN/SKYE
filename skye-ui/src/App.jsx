import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, ReferenceDot
} from 'recharts';
const styles = {
  container: {
    fontFamily: 'Inter, system-ui, sans-serif',
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%)',
    color: '#0a1628',
    padding: '24px',
    boxSizing: 'border-box',
    overflowX: 'hidden'
  },
  navbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 24px',
    background: 'rgba(255, 255, 255, 0.6)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.8)',
    borderRadius: '16px',
    boxShadow: '0 8px 32px rgba(31, 38, 135, 0.1)',
    marginBottom: '24px'
  },
  logoText: {
    fontSize: '24px',
    fontWeight: '800',
    color: '#2563eb',
    margin: 0,
    letterSpacing: '1px'
  },
  navInfo: {
    display: 'flex',
    gap: '32px',
    fontSize: '14px',
    fontWeight: '500',
    color: '#475569'
  },
  navClock: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#0a1628',
    fontVariantNumeric: 'tabular-nums'
  },
  statsBar: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '24px',
    marginBottom: '24px'
  },
  glassCard: {
    background: 'rgba(255, 255, 255, 0.6)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.8)',
    borderRadius: '16px',
    boxShadow: '0 4px 16px rgba(31, 38, 135, 0.05)',
    padding: '20px'
  },
  mainLayout: {
    display: 'grid',
    gridTemplateColumns: '300px 1fr 350px',
    gap: '24px',
    alignItems: 'start'
  },
  sidebar: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px'
  },
  subsystemCard: {
    background: 'rgba(255, 255, 255, 0.7)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.8)',
    borderRadius: '12px',
    padding: '16px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.02)',
    transition: 'all 0.3s ease'
  },
  flexBetween: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  graphsArea: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px'
  },
  graphCard: {
    background: 'rgba(255, 255, 255, 0.6)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.8)',
    borderRadius: '16px',
    padding: '20px',
    boxShadow: '0 4px 16px rgba(31, 38, 135, 0.05)',
    height: '340px'
  },
  alertsArea: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px'
  },
  faultCard: {
    background: 'linear-gradient(135deg, rgba(254, 226, 226, 0.8) 0%, rgba(255, 255, 255, 0.6) 100%)',
    backdropFilter: 'blur(20px)',
    border: '1px solid #fecaca',
    borderRadius: '12px',
    padding: '16px',
    boxShadow: '0 4px 16px rgba(239, 68, 68, 0.1)'
  },
  badge: {
    padding: '4px 10px',
    borderRadius: '999px',
    fontSize: '12px',
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  },
  btn: {
    padding: '8px 16px',
    borderRadius: '8px',
    border: 'none',
    fontWeight: '600',
    cursor: 'pointer',
    fontSize: '14px',
    transition: 'all 0.2s',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px'
  },
  btnPrimary: {
    background: '#2563eb',
    color: '#fff',
    boxShadow: '0 4px 12px rgba(37, 99, 235, 0.2)'
  },
  btnOutline: {
    background: 'rgba(255,255,255,0.5)',
    border: '1px solid #cbd5e1',
    color: '#334155'
  },
  modalOverlay: {
    position: 'fixed',
    top: 0, left: 0, right: 0, bottom: 0,
    background: 'rgba(15, 23, 42, 0.3)',
    backdropFilter: 'blur(8px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 9999
  },
  modalCard: {
    background: 'rgba(255, 255, 255, 0.9)',
    border: '1px solid rgba(255,255,255,1)',
    borderRadius: '24px',
    padding: '32px',
    width: '100%',
    maxWidth: '560px',
    boxShadow: '0 24px 48px rgba(0,0,0,0.1)'
  }
};
const norm = (val, normal, fault, drop = false) => {
  const range = Math.abs(fault - normal);
  if (drop) {
    return 50 + ((normal - val) * (50 / range));
  }
  return 50 + ((val - normal) * (50 / range));
};
const generateDataPoint = (tick, phase) => {
  const noise = () => (Math.random() - 0.5);
  const sin = Math.sin(tick / 5);
  const cos = Math.cos(tick / 7);
  let engVibration = 1.0 + sin * 0.1 + noise() * 0.05;
  let engEGT = 650 + cos * 15 + noise() * 5;
  let engOil = 60 + noise() * 2;
  let engN1 = 90 + sin * 1 + noise() * 0.5;
  if (tick >= 60) {
    const severity = Math.min(1, (tick - 60) / 10);
    engVibration += severity * 2.1;
    engEGT += severity * 120;
  }
  let structStress = 0.4 + sin * 0.05 + noise() * 0.02;
  let structCycles = 1200 + tick;
  let structAccel = 0.035 + cos * 0.01 + noise() * 0.005;
  if (tick >= 45) {
    if (tick === 45) structStress = 0.91;
    else structStress = 0.85 + noise() * 0.05;
  }
  let hydPressure = 3025 + noise() * 20;
  let hydActuator = 130 + sin * 5 + noise() * 2;
  if (tick >= 70) {
    const severity = Math.min(1, (tick - 70) / 15);
    hydPressure -= severity * 375;
  }
  let ecsCabin = 11.75 + noise() * 0.1;
  let ecsBleed = 210 + cos * 5 + noise() * 2;

  return {
    tick,
    engVibration, engEGT, engOil, engN1,
    structStress, structCycles, structAccel,
    hydPressure, hydActuator,
    ecsCabin, ecsBleed,
    engVibration_n: norm(engVibration, 1.0, 2.5),
    engEGT_n: norm(engEGT, 650, 750),
    engOil_n: norm(engOil, 60, 40, true),
    engN1_n: norm(engN1, 90, 80, true),

    structStress_n: norm(structStress, 0.4, 0.8),
    structAccel_n: norm(structAccel, 0.035, 0.15),

    hydPressure_n: norm(hydPressure, 3025, 2800, true),
    hydActuator_n: norm(hydActuator, 130, 200),

    ecsCabin_n: norm(ecsCabin, 11.75, 10.5, true),
    ecsBleed_n: norm(ecsBleed, 210, 260)
  };
};

export default function SkyeDashboard() {
  const [data, setData] = useState([]);
  const [isPlaying, setIsPlaying] = useState(true);
  const [tick, setTick] = useState(0);
  const [activeFaults, setActiveFaults] = useState([]);
  const [flightPhase, setFlightPhase] = useState('Cruise');
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const [modalFault, setModalFault] = useState(null);

  const graphRefs = {
    eng: useRef(null),
    struct: useRef(null),
    hyd: useRef(null),
    ecs: useRef(null)
  };
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!isPlaying) return;
    if (tick >= 100) return;

    const interval = setInterval(() => {
      const pt = generateDataPoint(tick, flightPhase);
      setData(prev => [...prev, pt]);
      setTick(t => t + 1);
      if (pt.engVibration > 2.5 && !activeFaults.find(f => f.id === 'eng1')) {
        setActiveFaults(prev => [...prev, {
          id: 'eng1', sub: 'ENG-1', name: 'Bearing Degradation', sev: 'CRITICAL',
          tick: pt.tick, val: pt.engVibration_n, rul: 'Est. 140 flight hrs to inspection',
          desc: 'The engine bearing is showing abnormal vibration — it is wearing out faster than expected',
          impact: 'If ignored, the bearing can seize, causing engine failure',
          graphEv: `Notice the red dot at second ${pt.tick} — vibration jumped suddenly`,
          action: 'Schedule bearing inspection within next 2 flights',
          ref: 'eng'
        }]);
      }
      
      if (pt.structStress > 0.8 && !activeFaults.find(f => f.id === 'struct1')) {
        setActiveFaults(prev => [...prev, {
          id: 'struct1', sub: 'AIRFRAME', name: 'Micro-Fracture Detected', sev: 'WARNING',
          tick: pt.tick, val: pt.structStress_n, rul: 'Est. 280 flight hrs',
          desc: 'A localized stress index spike indicates a potential micro-fracture in the airframe',
          impact: 'Repeated stress can propagate the fracture, leading to structural fatigue',
          graphEv: `Notice the red dot at second ${pt.tick} — stress index spiked to ${pt.structStress.toFixed(2)}`,
          action: 'Perform NDT (Non-Destructive Testing) during next A-check',
          ref: 'struct'
        }]);
      }

      if (pt.hydPressure < 2800 && !activeFaults.find(f => f.id === 'hyd1')) {
        setActiveFaults(prev => [...prev, {
          id: 'hyd1', sub: 'HYD-SYS', name: 'Hydraulic Pressure Decay', sev: 'CRITICAL',
          tick: pt.tick, val: pt.hydPressure_n, rul: 'Est. 60 flight hrs',
          desc: 'System fluid pressure is steadily decaying below safe operational thresholds',
          impact: 'Loss of pressure could lead to sluggish flight control surface actuation',
          graphEv: `Notice the red dot at second ${pt.tick} — pressure dropped below 2800 PSI`,
          action: 'Check for fluid leaks in the main accumulator and top-up immediately',
          ref: 'hyd'
        }]);
      }
    }, 800);

    return () => clearInterval(interval);
  }, [isPlaying, tick, flightPhase, activeFaults]);

  const resetSim = () => {
    setData([]);
    setTick(0);
    setActiveFaults([]);
    setIsPlaying(true);
  };

  const getHealth = (sub) => {
    const f = activeFaults.find(f => f.id.startsWith(sub));
    if (!f) return { score: sub === 'eng'?94:sub==='struct'?88:sub==='hyd'?96:99, status: 'HEALTHY', color: '#10b981' };
    if (f.sev === 'CRITICAL') return { score: sub === 'eng'?61:58, status: 'CRITICAL', color: '#ef4444' };
    return { score: 72, status: 'WARNING', color: '#f59e0b' };
  };

  const engH = getHealth('eng');
  const structH = getHealth('struct');
  const hydH = getHealth('hyd');
  const ecsH = getHealth('ecs');

  const avgHealth = Math.round((engH.score + structH.score + hydH.score + ecsH.score) / 4);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: 'rgba(255,255,255,0.95)', border: '1px solid #e2e8f0', padding: '12px', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
          <p style={{ margin: '0 0 8px 0', fontWeight: 'bold' }}>Sec: {label}</p>
          {payload.map((p, i) => {
            const rawKey = p.dataKey.replace('_n', '');
            const rawVal = data.find(d => d.tick === label)?.[rawKey];
            return (
              <div key={i} style={{ color: p.color, fontSize: '12px', marginBottom: '4px' }}>
                {rawKey}: {rawVal?.toFixed(2)}
              </div>
            );
          })}
        </div>
      );
    }
    return null;
  };

  const scrollToGraph = (refName) => {
    graphRefs[refName].current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  return (
    <div style={styles.container}>
      <nav style={styles.navbar}>
        <div style={{display: 'flex', alignItems: 'center', gap: '16px'}}>
          <h1 style={styles.logoText}>SKYE</h1>
          <div style={{padding: '4px 8px', background: '#e0e7ff', color: '#3730a3', borderRadius: '6px', fontSize: '12px', fontWeight: 'bold'}}>EDGE AI DIAGNOSTICS</div>
        </div>
        <div style={styles.navInfo}>
          <span>Flight: SQ-471</span>
          <span>Aircraft: A320-NEO</span>
          <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
            Phase: 
            <select value={flightPhase} onChange={e=>setFlightPhase(e.target.value)} style={{background: 'transparent', border: '1px solid #cbd5e1', borderRadius: '4px', padding: '2px 8px'}}>
              <option>Takeoff</option>
              <option>Cruise</option>
              <option>Landing</option>
            </select>
            at 38,000ft
          </div>
        </div>
        <div style={styles.navClock}>
          {currentTime} <span style={{color: '#ef4444', marginLeft: '8px'}}>• LIVE</span>
        </div>
      </nav>
      <div style={styles.statsBar}>
        <div style={styles.glassCard}>
          <div style={{fontSize: '12px', color: '#64748b', textTransform: 'uppercase', fontWeight: '600'}}>Total Anomalies</div>
          <div style={{fontSize: '32px', fontWeight: '700', color: '#0f172a'}}>{activeFaults.length}</div>
        </div>
        <div style={styles.glassCard}>
          <div style={{fontSize: '12px', color: '#64748b', textTransform: 'uppercase', fontWeight: '600'}}>Active Faults</div>
          <div style={{fontSize: '32px', fontWeight: '700', color: activeFaults.length>0?'#ef4444':'#0f172a'}}>{activeFaults.length}</div>
        </div>
        <div style={styles.glassCard}>
          <div style={{fontSize: '12px', color: '#64748b', textTransform: 'uppercase', fontWeight: '600'}}>Flight Hrs Monitored</div>
          <div style={{fontSize: '32px', fontWeight: '700', color: '#0f172a'}}>{(12430 + tick/100).toFixed(1)}</div>
        </div>
        <div style={styles.glassCard}>
          <div style={{fontSize: '12px', color: '#64748b', textTransform: 'uppercase', fontWeight: '600'}}>Fleet Health Avg</div>
          <div style={{fontSize: '32px', fontWeight: '700', color: avgHealth<80?'#f59e0b':'#10b981'}}>{avgHealth}%</div>
        </div>
      </div>

      <div style={styles.mainLayout}>
        <div style={styles.sidebar}>
          <div style={{...styles.glassCard, padding: '16px', display: 'flex', gap: '8px'}}>
            <button style={{...styles.btn, ...(isPlaying?styles.btnOutline:styles.btnPrimary), flex: 1}} onClick={() => setIsPlaying(!isPlaying)}>
              {isPlaying ? '⏸ Pause' : '▶ Resume'}
            </button>
            <button style={{...styles.btn, ...styles.btnOutline, flex: 1}} onClick={resetSim}>
              🔄 Reset
            </button>
          </div>

          {[
            { id: 'eng', name: 'ENGINE (ENG-1)', icon: '⚙️', h: engH },
            { id: 'struct', name: 'STRUCTURAL', icon: '🏗️', h: structH },
            { id: 'hyd', name: 'HYDRAULIC', icon: '💧', h: hydH },
            { id: 'ecs', name: 'ENVIRONMENTAL', icon: '🌡️', h: ecsH }
          ].map(sub => (
            <div key={sub.id} style={styles.subsystemCard}>
              <div style={{...styles.flexBetween, marginBottom: '12px'}}>
                <span style={{fontWeight: '700', fontSize: '14px', color: '#334155'}}>{sub.icon} {sub.name}</span>
                <span style={{...styles.badge, background: sub.h.color+'20', color: sub.h.color}}>
                  <span style={{display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: sub.h.color, marginRight: '6px', animation: sub.h.status!=='HEALTHY'?'pulse 1s infinite':''}}></span>
                  {sub.h.status}
                </span>
              </div>
              <div style={styles.flexBetween}>
                <span style={{fontSize: '28px', fontWeight: '800', color: sub.h.color}}>{sub.h.score}</span>
                <span style={{fontSize: '12px', color: '#94a3b8'}}>Health Score</span>
              </div>
            </div>
          ))}
        </div>
        <div style={styles.graphsArea}>
          
          <div style={styles.graphCard} ref={graphRefs.eng}>
            <h3 style={{margin: '0 0 16px 0', fontSize: '16px', color: '#1e293b'}}>⚙️ Engine (ENG-1) Parameters</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={data} margin={{top:10, right:30, left:0, bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="tick" stroke="#94a3b8" fontSize={12} />
                <YAxis domain={[0, 150]} hide />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="4 4" label={{position: 'insideTopLeft', value: 'FAULT THRESHOLD', fill: '#ef4444', fontSize: 10}} />
                <Line type="monotone" dataKey="engVibration_n" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="engEGT_n" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="engOil_n" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="engN1_n" stroke="#8b5cf6" strokeWidth={2} dot={false} isAnimationActive={false} />
                {activeFaults.find(f => f.id === 'eng1') && (
                  <ReferenceDot x={activeFaults.find(f=>f.id==='eng1').tick} y={activeFaults.find(f=>f.id==='eng1').val} r={6} fill="#ef4444" stroke="#fff" strokeWidth={2} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div style={styles.graphCard} ref={graphRefs.struct}>
            <h3 style={{margin: '0 0 16px 0', fontSize: '16px', color: '#1e293b'}}>🏗️ Structural (Airframe) Parameters</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={data} margin={{top:10, right:30, left:0, bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="tick" stroke="#94a3b8" fontSize={12} />
                <YAxis domain={[0, 150]} hide />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="structStress_n" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="structAccel_n" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                {activeFaults.find(f => f.id === 'struct1') && (
                  <ReferenceDot x={activeFaults.find(f=>f.id==='struct1').tick} y={activeFaults.find(f=>f.id==='struct1').val} r={6} fill="#ef4444" stroke="#fff" strokeWidth={2} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div style={styles.graphCard} ref={graphRefs.hyd}>
            <h3 style={{margin: '0 0 16px 0', fontSize: '16px', color: '#1e293b'}}>💧 Hydraulic (HYD-SYS) Parameters</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={data} margin={{top:10, right:30, left:0, bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="tick" stroke="#94a3b8" fontSize={12} />
                <YAxis domain={[0, 150]} hide />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="hydPressure_n" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="hydActuator_n" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                {activeFaults.find(f => f.id === 'hyd1') && (
                  <ReferenceDot x={activeFaults.find(f=>f.id==='hyd1').tick} y={activeFaults.find(f=>f.id==='hyd1').val} r={6} fill="#ef4444" stroke="#fff" strokeWidth={2} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div style={styles.graphCard} ref={graphRefs.ecs}>
            <h3 style={{margin: '0 0 16px 0', fontSize: '16px', color: '#1e293b'}}>🌡️ Environmental (ECS) Parameters</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={data} margin={{top:10, right:30, left:0, bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="tick" stroke="#94a3b8" fontSize={12} />
                <YAxis domain={[0, 150]} hide />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="ecsCabin_n" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="ecsBleed_n" stroke="#8b5cf6" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

        </div>
        <div style={styles.alertsArea}>
          <div style={{...styles.glassCard, padding: '16px', background: 'rgba(255,255,255,0.8)'}}>
            <h3 style={{margin: '0 0 16px 0', fontSize: '16px'}}>🚨 Active Faults</h3>
            {activeFaults.length === 0 ? (
              <div style={{color: '#64748b', fontSize: '14px', fontStyle: 'italic'}}>No active faults detected. All systems nominal.</div>
            ) : (
              <div style={{display: 'flex', flexDirection: 'column', gap: '12px'}}>
                {activeFaults.map(f => (
                  <div key={f.id} style={styles.faultCard}>
                    <div style={{...styles.flexBetween, marginBottom: '8px'}}>
                      <span style={{fontSize: '12px', fontWeight: '700', color: '#64748b'}}>{f.sub}</span>
                      <span style={{...styles.badge, background: f.sev==='CRITICAL'?'#ef4444':'#f59e0b', color: '#fff'}}>{f.sev}</span>
                    </div>
                    <div style={{fontSize: '16px', fontWeight: '700', color: '#0f172a', marginBottom: '4px'}}>{f.name}</div>
                    <div style={{fontSize: '12px', color: '#475569', marginBottom: '12px'}}>Detected at Sec: {f.tick}</div>
                    <div style={{fontSize: '12px', fontWeight: '600', color: '#b91c1c', marginBottom: '16px'}}>⏱️ {f.rul}</div>
                    <div style={{display: 'flex', gap: '8px'}}>
                      <button style={{...styles.btn, ...styles.btnPrimary, fontSize: '12px', padding: '6px 12px'}} onClick={() => setModalFault(f)}>
                        💡 Explain
                      </button>
                      <button style={{...styles.btn, ...styles.btnOutline, fontSize: '12px', padding: '6px 12px'}} onClick={() => scrollToGraph(f.ref)}>
                        🔍 Locate on Graph
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      {modalFault && (
        <div style={styles.modalOverlay} onClick={() => setModalFault(null)}>
          <div style={styles.modalCard} onClick={e => e.stopPropagation()}>
            <div style={{...styles.flexBetween, borderBottom: '1px solid #e2e8f0', paddingBottom: '16px', marginBottom: '24px'}}>
              <h2 style={{margin: 0, color: '#ef4444', fontSize: '24px'}}>{modalFault.name}</h2>
              <button style={{background: 'transparent', border: 'none', fontSize: '24px', cursor: 'pointer'}} onClick={() => setModalFault(null)}>×</button>
            </div>
            
            <div style={{display: 'flex', flexDirection: 'column', gap: '20px'}}>
              <div>
                <h4 style={{margin: '0 0 8px 0', color: '#334155', display: 'flex', alignItems: 'center', gap: '8px'}}>⚠️ What happened</h4>
                <p style={{margin: 0, color: '#475569', lineHeight: '1.5'}}>{modalFault.desc}</p>
              </div>
              <div>
                <h4 style={{margin: '0 0 8px 0', color: '#334155', display: 'flex', alignItems: 'center', gap: '8px'}}>🛑 Why it matters</h4>
                <p style={{margin: 0, color: '#475569', lineHeight: '1.5'}}>{modalFault.impact}</p>
              </div>
              <div>
                <h4 style={{margin: '0 0 8px 0', color: '#334155', display: 'flex', alignItems: 'center', gap: '8px'}}>📈 What the graph shows</h4>
                <p style={{margin: 0, color: '#475569', lineHeight: '1.5'}}>{modalFault.graphEv}</p>
              </div>
              <div style={{background: '#f0fdf4', border: '1px solid #bbf7d0', padding: '16px', borderRadius: '8px'}}>
                <h4 style={{margin: '0 0 8px 0', color: '#166534', display: 'flex', alignItems: 'center', gap: '8px'}}>✅ Recommended action</h4>
                <p style={{margin: 0, color: '#15803d', lineHeight: '1.5', fontWeight: '500'}}>{modalFault.action}</p>
              </div>
            </div>
          </div>
        </div>
      )}
      <style>{`
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(0,0,0, 0.4); }
          70% { box-shadow: 0 0 0 6px rgba(0,0,0, 0); }
          100% { box-shadow: 0 0 0 0 rgba(0,0,0, 0); }
        }
      `}</style>
    </div>
  );
}
